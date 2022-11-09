import copy
import sys

sys.path.append("..")
import torch
import numpy as np
from collections import deque, namedtuple

from sabre360_with_qoe import Session, LogFile, LogFileForTraining
from view_prediction import TestPrediction
from utils import get_trace_file, get_tiles_in_viewport, Pose2VideoXY
from abr.myABR import TestAbr

# todo: set STATE_DIMENSION = ? & ACTION_DIMENSION = ?
STATE_DIMENSION = 27
ACTION_DIMENSION = 27
HISTORY_LENGTH = 1
MAX_ET_LEN = 5  # s     # todo: 确定最大的ET buffer长度
TPUT_HISTORY_LEN = 10
TILES_X = 8
TILES_Y = 8
BITRATE_LEVEL = 6

SLEEP_PERIOD = 500  # 暂停时长 ms

# ============= Config Setting =============
default_config = {}
default_config['ewma_half_life'] = [4, 1]  # seconds
default_config['buffer_size'] = 5  # seconds
default_config['log_file'] = 'log/session.log'
default_config['abr'] = TestAbr  # 不调用

TiledAction = namedtuple('TiledAction', 'segment tile quality delay')

OUTPUT_LOG = False


class RLEnv:
    def __init__(self):
        self.state_dim = STATE_DIMENSION
        self.action_dim = ACTION_DIMENSION
        self.state = torch.zeros(HISTORY_LENGTH, STATE_DIMENSION)

        # ============= config 配置 =============
        self.net_id, self.video_id, self.user_id = 0, 0, 0
        network_file, video_file, user_file = get_trace_file(self.net_id, self.video_id, self.user_id)

        self.config = default_config.copy()
        self.config['bandwidth_trace'] = network_file
        self.config['manifest'] = video_file
        self.config['pose_trace'] = user_file

        # ============= 新建传输任务 =============
        self.session = Session(self.config)
        if not OUTPUT_LOG:  # 禁用日志输出
            self.session.log_file = LogFileForTraining(self.session.session_info, self.config['log_file'])
        self.manifest = self.session.session_info.get_manifest()
        self.video_size = self.manifest.segments  # [segment][tile][level]
        self.video_time = self.session.manifest.segment_duration * len(self.session.manifest.segments)
        self.bitrate_level = self.session.manifest.bitrates

        # 显式明确视角预测算法
        self.session.viewport_prediction = TestPrediction(self.session.session_info, self.session.session_events)
        self.session.session_info.set_viewport_predictor(self.session.viewport_prediction)

        # ============= state相关变量 =============
        self.past_k_tput = []
        self.avg_level_dict = {}
        self.base_buffer_depth = 0
        self.enhance_buffer_depth = 0
        self.latest_BT_segment = -1
        self.latest_ET_segment = -1
        self.pred_view_dict = {}
        self.pred_tiles_dict = {}
        self.contents = {}
        return

    def reset(self, network_trace_id=0, video_trace_id=0, user_trace_id=0):
        # ============= config配置 =============
        self.net_id = network_trace_id
        self.video_id = video_trace_id
        self.user_id = user_trace_id
        network_file, video_file, user_file = get_trace_file(self.net_id, self.video_id, self.user_id)

        self.config = default_config.copy()
        self.config['bandwidth_trace'] = network_file
        self.config['manifest'] = video_file
        self.config['pose_trace'] = user_file

        # ============= 新建传输任务 =============
        self.session = Session(self.config)
        if not OUTPUT_LOG:  # 禁用日志输出
            self.session.log_file = LogFileForTraining(self.session.session_info, self.config['log_file'])
        self.manifest = self.session.session_info.get_manifest()
        self.video_size = self.manifest.segments  # [segment][tile][level]
        self.video_time = self.session.manifest.segment_duration * len(self.session.manifest.segments)
        self.bitrate_level = self.session.manifest.bitrates

        # 显式明确视角预测算法
        self.session.viewport_prediction = TestPrediction(self.session.session_info, self.session.session_events)
        self.session.session_info.set_viewport_predictor(self.session.viewport_prediction)

        # ============= state相关变量初始化 =============
        self.past_k_tput.clear()
        self.avg_level_dict.clear()
        self.base_buffer_depth = 0
        self.enhance_buffer_depth = 0
        self.latest_BT_segment = -1
        self.latest_ET_segment = -1
        self.pred_view_dict.clear()
        self.pred_tiles_dict.clear()
        self.contents.clear()

        # 吞吐量
        s_throughput = self.past_k_tput.copy()
        while len(s_throughput) < 10:
            s_throughput.insert(0, 10000)
        assert len(s_throughput) == 10
        s_throughput = np.array(s_throughput) / 10000  # 归一化
        self.state[0, 0:10] = torch.Tensor(s_throughput)

        # 待下载的数据量
        first_segment = 0
        self.update_viewport_pred(first_segment, first_segment)
        future_size_in_BT, future_size_in_ET = self.update_download_size(first_segment, self.contents)
        s_future_size = future_size_in_BT + future_size_in_ET
        assert len(s_future_size) == 1 + MAX_ET_LEN
        s_future_size = np.array(s_future_size) / 1000000.
        self.state[0, 13:19] = torch.Tensor(s_future_size)

        # 平均码率等级
        start_et_segment = 0
        end_et_segment = start_et_segment + MAX_ET_LEN
        s_avg_level = []
        for segment_idx in range(start_et_segment, end_et_segment + 1):
            s_avg_level.append(0)
        assert len(s_avg_level) == MAX_ET_LEN + 1
        self.state[0, 19:25] = torch.Tensor(s_avg_level)

        # 视点运动信息
        std_x, std_y = self.calculate_pose_spead(first_segment, 0)
        self.state[0, 25] = std_x * 100
        self.state[0, 26] = std_y * 100

        return self.state

    def step(self, ppo_output):
        one_step_reward, done = 0, False
        is_BT_download = False
        is_ET_download = False
        is_sleep = False
        action = []
        if ppo_output == ACTION_DIMENSION - 1:
            ''' delay '''
            action.append(TiledAction(-1, -1, -1, SLEEP_PERIOD))
            is_sleep = True
        elif ppo_output == ACTION_DIMENSION - 2:
            ''' BT '''
            is_BT_download = True
            segment_id = self.latest_BT_segment + 1
            assert segment_id <= len(self.session.manifest.segments) - 1
            for tile_id in range(TILES_X * TILES_Y):
                action.append(TiledAction(segment_id, tile_id, 0, 0))
        else:
            ''' ET '''
            is_ET_download = True
            start_idx = self.session.buffer.get_played_segments()
            offset = self.session.buffer.get_played_segment_partial()
            if offset != 0:
                start_idx += 1
            segment_id = int(start_idx + ppo_output // (BITRATE_LEVEL - 1))
            assert segment_id < self.latest_BT_segment + 1
            assert segment_id <= len(self.session.manifest.segments) - 1
            level = int(1 + ppo_output % (BITRATE_LEVEL - 1))

            tiles_list = self.pred_tiles_dict[segment_id]
            download_tile = tiles_list.copy()

            downloaded_tile_info = []
            if segment_id in self.contents:
                downloaded_tile_info = self.contents[segment_id]

            if len(downloaded_tile_info) == 0:
                for tile_id in tiles_list:
                    action.append(TiledAction(segment_id, tile_id, level, 0))
            else:
                for i in range(len(tiles_list)):
                    if downloaded_tile_info[tiles_list[i]] != 0:  # 不重复下载更新过的tile
                        download_tile.remove(tiles_list[i])
                assert len(download_tile) != 0
                for tile_id in download_tile:
                    action.append(TiledAction(segment_id, tile_id, level, 0))
        # print(action)
        ''' ================== 和环境交互 ===================== '''
        self.stimulator(action)  # 调用stimulator模拟播放&下载
        play_head = self.session.session_info.buffer.get_play_head()  # ms
        if play_head >= self.video_time:  # 当前视频播放完成
            done = True
        if action is None:
            return self.state, 0, done

        ''' ================== 更新缓冲区状态 ================== '''
        buffers = self.session.session_info.get_buffer()
        playhead = buffers.get_play_head()
        first_segment = buffers.get_played_segments()
        first_segment_offset = buffers.get_played_segment_partial()
        segment_depth = buffers.get_buffer_depth()
        self.contents = {}  # 每个segment内的tile信息，未被下载的是None
        for i in range(segment_depth):
            segment_idx = first_segment + i
            self.contents[segment_idx] = buffers.get_buffer_contents(segment_idx)
            # print(segment_idx, " : ", self.contents[segment_idx])

        '''####################################################
        # 1. 更新过去k时刻吞吐量
        # 2. 更新缓冲区长度
        # 3. 更新实时的视角预测结果
        # 4. 更新待下载的数据量
        # 5. 更新buffer视窗内tile的平均码率等级 (根据最新视窗预测结果)
        #######################################################
        '''
        first_et_segment = first_segment
        if first_segment_offset != 0:
            first_et_segment += 1

        if not is_sleep:  # 仅在不暂停下载时更新
            self.update_throughput(action)
        self.update_buffer_length(action, is_BT_download, is_ET_download, first_segment_offset)
        self.update_viewport_pred(first_segment, first_et_segment)
        future_size_in_BT, future_size_in_ET = self.update_download_size(first_et_segment, self.contents)
        self.update_avg_level(first_segment, first_et_segment, self.contents)

        ''' ================== 计算state ================== '''
        # 吞吐量
        s_throughput = self.past_k_tput.copy()
        while len(s_throughput) < 10:
            s_throughput.insert(0, 10000)
        assert len(s_throughput) == 10
        s_throughput = np.array(s_throughput) / 10000  # 归一化
        self.state[0, 0:10] = torch.Tensor(s_throughput)

        # buffer长度
        self.state[0, 10] = self.base_buffer_depth
        self.state[0, 11] = self.enhance_buffer_depth

        # 当前播放时间
        self.state[0, 12] = playhead / 1000.  # s

        # 待下载的数据量
        s_future_size = future_size_in_BT + future_size_in_ET
        assert len(s_future_size) == 1 + MAX_ET_LEN
        s_future_size = np.array(s_future_size) / 1000000.
        self.state[0, 13:19] = torch.Tensor(s_future_size)

        # 平均码率等级
        end_et_segment = first_et_segment + MAX_ET_LEN
        s_avg_level = []
        for segment_idx in range(first_et_segment, end_et_segment + 1):
            if segment_idx in self.avg_level_dict:
                s_avg_level.append(self.avg_level_dict[segment_idx])
            else:
                s_avg_level.append(0)
        assert len(s_avg_level) == MAX_ET_LEN + 1
        self.state[0, 19:25] = torch.Tensor(s_avg_level)

        # 视点运动信息
        std_x, std_y = self.calculate_pose_spead(first_segment, first_segment_offset)
        self.state[0, 25] = std_x * 100
        self.state[0, 26] = std_y * 100

        # print('play_head = ', play_head)
        # print('video_time = ', self.video_time)
        # print('download time = ', self.session.total_download_time)
        # print('base_buffer_depth = ', self.base_buffer_depth)
        # print('qoe_one_step = ', self.session.qoe_one_step, '\n')
        return self.state, self.session.score_one_step, done

    def update_throughput(self, action):
        ''' 更新历史带宽信息 '''
        size = 0
        for i in range(len(action)):
            size += self.video_size[action[i].segment][action[i].tile][action[i].quality]  # bit
        download_time = self.session.total_download_time  # ms
        assert download_time != 0
        tput = size / download_time  # kbps
        if len(self.past_k_tput) >= TPUT_HISTORY_LEN:
            self.past_k_tput.pop(0)
        self.past_k_tput.append(tput)

    def update_buffer_length(self, action, is_BT_download, is_ET_download, first_segment_offset):
        is_BT_download_re = False  # todo：可以删掉
        ''' 更新ET和BT buffer的长度 '''
        if action[0].segment > self.latest_BT_segment:
            is_BT_download_re = True
        assert is_BT_download_re == is_BT_download

        self.latest_BT_segment = max(action[0].segment, self.latest_BT_segment)
        self.base_buffer_depth = self.session.buffer.get_buffer_depth() - first_segment_offset / 1000.  # 基础层长度即缓冲区实际长度

        if is_ET_download:  # ET download
            self.latest_ET_segment = max(self.latest_ET_segment, action[0].segment)  # 增强层长度需要根据实际的更新决策定义
        self.enhance_buffer_depth = self.base_buffer_depth - (self.latest_BT_segment - self.latest_ET_segment)

    def update_viewport_pred(self, first_segment, first_et_segment):
        ''' 更新视角预测结果 (对buffer内所有tile) '''
        # 1. 执行一次视角预测
        start_segment_idx = first_et_segment  # 确定ET层可下载的范围 start~end（当前播放视频也预测,end是最后一个可下载的segment）
        end_segment_idx = start_segment_idx + MAX_ET_LEN
        pred_view_dict = {}
        model_x, model_y = self.session.session_info.get_viewport_predictor().build_model(end_segment_idx)
        if model_x is None:
            for seg_idx in range(first_segment, end_segment_idx + 1):
                pred_view_dict[seg_idx] = [0.5, 0.5]  # 默认视点 (0.5,0.5)
        else:
            for seg_idx in range(first_segment, end_segment_idx + 1):
                pred_view = self.session.session_info.get_viewport_predictor().predict_view(model_x, model_y, seg_idx)
                (x_pred, y_pred) = pred_view
                pred_view_dict[seg_idx] = [x_pred, y_pred]
        # 2 根据预测结果确定视窗内的tile
        self.pred_tiles_dict = {}
        for seg_idx in range(first_segment, end_segment_idx + 1):
            x_pred = pred_view_dict[seg_idx][0]
            y_pred = pred_view_dict[seg_idx][0]
            tiles_in_viewport = get_tiles_in_viewport(x_pred, y_pred)
            self.pred_tiles_dict[seg_idx] = tiles_in_viewport

    def update_download_size(self, first_et_segment, contents):
        ''' 更新待下载的数据量 '''
        # ======> Part 1: segment in BT buffer - 最低质量全画幅下载
        size_in_BT = []
        next_BT_segment = self.latest_BT_segment + 1
        download_bit = 0
        if next_BT_segment <= len(self.video_size) - 1:
            for i in range(len(self.video_size[next_BT_segment])):
                download_bit += self.video_size[next_BT_segment][i][0]
        else:
            download_bit = 0
        size_in_BT.append(download_bit)

        # ======> Part 2: segments in ET buffer - 待更新码率
        size_in_ET = []
        for i in range(MAX_ET_LEN):  # 遍历et buffer中的segment
            download_bit = 0
            segment_idx = first_et_segment + i
            pred_tiles = self.pred_tiles_dict[segment_idx]
            if segment_idx >= len(self.video_size) - 1:  # 超过segment idx
                size_in_ET.append(download_bit)
                continue

            if segment_idx <= self.latest_ET_segment:
                # 更新ET segment的情况
                downloaded_tile_info = contents[segment_idx]  # 已下载的tile信息
                for tile in pred_tiles:  # 下载未被下载过的tile
                    if downloaded_tile_info[tile] == 0:
                        download_bit += self.video_size[segment_idx][tile][len(self.bitrate_level) - 1]
            else:
                # 新下载ET segment的情况
                for tile in pred_tiles:  # 下载全部预测视窗内的tile
                    download_bit += self.video_size[segment_idx][tile][len(self.bitrate_level) - 1]
            size_in_ET.append(download_bit)
        assert len(size_in_ET) == MAX_ET_LEN
        return size_in_BT, size_in_ET  # bits

    def update_avg_level(self, first_segment, first_et_segment, contents):
        ''' 更新ET(以及ET前一个)中视窗内所有tile的平均码率等级 '''
        start_segment_idx = first_et_segment  # 确定ET层可下载的范围 start~end（当前播放视频也预测,end是最后一个可下载的segment）
        end_segment_idx = start_segment_idx + MAX_ET_LEN
        for segment_idx in range(first_segment, end_segment_idx + 1):  # 遍历buffer内每个chunk
            pred_tiles = self.pred_tiles_dict[segment_idx]
            sum_level = 0
            none_cnt = 0
            if segment_idx > self.latest_BT_segment:  # 缓冲区内没有这个seg，则平均码率等级记为-1
                self.avg_level_dict[segment_idx] = -1
                continue
            assert segment_idx in contents
            for tile in pred_tiles:
                downloaded_tile_info = contents[segment_idx]
                tmp = downloaded_tile_info[tile]
                if tmp is None:
                    none_cnt += 1
                else:
                    sum_level += tmp
            avg_level = sum_level / max((len(pred_tiles) - none_cnt), 1)
            self.avg_level_dict[segment_idx] = avg_level

    def calculate_pose_spead(self, played_segment, offset):
        ''' 计算视点的运动速度(方差) '''
        if offset == 0:
            played_segment -= 1
        pose_trace = self.session.prev_pose_trace
        if len(pose_trace) <= 3:
            return 0, 0
        # 1. 取最新的5个值
        pose_num = min(len(pose_trace), 5)
        trace_x, trace_y = [], []
        for i in range(pose_num):
            assert played_segment - i in pose_trace
            pose = pose_trace[played_segment - i]
            video_x, video_y = Pose2VideoXY(pose)
            trace_x.append(video_x)
            trace_y.append(video_y)

        # 2. 对x轴坐标进行调整
        for i in range(1, len(trace_x)):
            if trace_x[i] - trace_x[i - 1] >= 0.5:
                trace_x[i] -= 1.0
            elif trace_x[i] - trace_x[i - 1] <= -0.5:
                trace_x[i] += 1.0
        trace_x = np.array(trace_x)
        trace_y = np.array(trace_y)

        # 3. 计算std
        std_x = np.std(trace_x)
        std_y = np.std(trace_y)
        return std_x, std_y

    def stimulator(self, action):
        ''' 和环境交互 '''
        # 获取最长播放距离
        view_log_begin = self.session.buffer.get_played_segments()
        view_log_end = view_log_begin + self.session.buffer.get_buffer_depth() + 1
        view_log_end = min(view_log_end, len(self.session.manifest.segments))

        delay = 0
        if action is None:
            # pause until the end of the current segment
            delay = self.session.manifest.segment_duration - self.session.session_info.buffer.get_played_segment_partial()
            # delay = video_time - self.session_info.buffer.get_play_head()
        elif action[0].delay is not None and action[0].delay > 0:
            delay = action[0].delay

        # 不设置最大buffer
        # if action is not None:
        #     buffer_end = (action[0].segment + 1) * self.session.manifest.segment_duration
        #     if delay < buffer_end - self.session.buffer.get_play_head() - self.session.buffer_size - 0.001:
        #         delay = buffer_end - self.session.buffer.get_play_head() - self.session.buffer_size

        if delay > 0:
            # shares self.consumed_download_time with self.consume_download_time()
            self.session.consumed_download_time = 0
            self.session.network_model.delay(delay)
            wall_time, stall_time = self.session.consume_download_time(delay, time_is_play_time=True)
            self.session.session_events.trigger_network_delay_event(delay)

            ''' 记录每一个播放过segment的pose trace'''
            cur_played_segment = self.session.buffer.get_played_segments()  # 正在播放的segment
            if self.session.buffer.get_played_segment_partial() == 0:  # 尚未开始播放
                cur_played_segment -= 1
            while cur_played_segment >= 0 and self.session.last_played_segment <= cur_played_segment:
                start_time = self.session.last_played_segment * self.manifest.segment_duration
                end_time = (self.session.last_played_segment + 1) * self.manifest.segment_duration
                pose_list = self.session.user_model.get_pose_in_qoe(start_time, end_time)
                self.session.prev_pose_trace[self.session.last_played_segment] = pose_list[0]
                self.session.last_played_segment += 1
            assert self.session.last_played_segment > cur_played_segment

            self.session.score_one_step = - 5. * stall_time
            self.session.total_score += self.session.score_one_step
            return

        self.session.total_download_time = 0  # 一次决策的总下载用时
        progress_list = []
        bandwidth_usage = 0
        for i in range(len(action)):
            if i == 0:
                is_first_tile = True
            else:
                is_first_tile = False

            size = self.session.manifest.segments[action[i].segment][action[i].tile][
                action[i].quality]  # 读取待下载的tile的size
            bandwidth_usage += size
            self.session.consumed_download_time = 0
            ''' 模拟下载过程 '''
            progress = self.session.network_model.download(size, action[i], is_first_tile, None)
            self.session.total_download_time += progress.time
            progress_list.append(copy.deepcopy(progress))

        ''' 模拟视频播放进程 '''
        wall_time, stall_time = self.session.consume_download_time(self.session.total_download_time)  # ms

        # todo：计算视角预测准确度
        ''' 记录每一个segment的pose trace'''
        cur_played_segment = self.session.buffer.get_played_segments()  # 正在播放的segment
        if self.session.buffer.get_played_segment_partial() == 0:  # 尚未开始播放
            cur_played_segment -= 1
        while cur_played_segment >= 0 and self.session.last_played_segment <= cur_played_segment:
            start_time = self.session.last_played_segment * self.manifest.segment_duration
            end_time = (self.session.last_played_segment + 1) * self.manifest.segment_duration
            pose_list = self.session.user_model.get_pose_in_qoe(start_time, end_time)
            self.session.prev_pose_trace[self.session.last_played_segment] = pose_list[0]
            self.session.last_played_segment += 1
        assert self.session.last_played_segment > cur_played_segment

        ''' QoE计算 '''
        segment_idx = action[0].segment
        download_tile = {}
        for i in range(len(action)):
            download_tile[action[i].tile] = action[i].quality

        # 1. 分项指标计算
        start_time = segment_idx * self.manifest.segment_duration
        end_time = (segment_idx + 1) * self.manifest.segment_duration
        pose_list = self.session.user_model.get_pose_in_qoe(start_time, end_time)
        assert len(pose_list) > 0
        bandwidth_usage /= 1024.
        # -----------------------------------------------------
        if self.session.buffer.get_play_head() <= segment_idx * self.session.manifest.segment_duration:
            delta_quality = self.session.calculate_delta_quality(pose_list, segment_idx, download_tile)
            delta_var_space = self.session.calculate_delta_var_space(pose_list, segment_idx, download_tile)
            delta_var_time = self.session.calculate_delta_var_time(segment_idx)
            bandwidth_wastage = self.session.calculate_bandwidth_wastage(pose_list, segment_idx, download_tile)
        else:  # 播放后才下载的质量不计入qoe
            delta_quality = 0
            delta_var_space = 0
            delta_var_time = 0
            bandwidth_wastage = bandwidth_usage
        assert bandwidth_wastage <= bandwidth_usage

        # 2. 线性组合
        self.session.qoe_one_step = delta_quality / 8 - 5. * stall_time - 0.5 * delta_var_space - 1. * delta_var_time - 0.5 * bandwidth_wastage / 8
        # self.session.qoe_one_step = delta_quality / 8 - 1.85 * stall_time - 0.5 * delta_var_space - 1 * delta_var_time - 0.5 * bandwidth_usage / 8
        self.session.total_score += self.session.score_one_step

        for i in range(len(action)):
            progress = progress_list[i]
            if progress.abandon is None:  # 没有放弃下载
                self.session.estimator.push(progress)
                self.session.buffer.put_in_buffer(progress.segment, progress.tile,
                                                  progress.quality)  # 将下载的chunk放入buffer
                self.session.log_file.log_download(progress)
        return

    def close(self):
        print("Environment closed.")

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim
