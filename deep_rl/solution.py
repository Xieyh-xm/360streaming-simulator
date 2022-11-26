import numpy as np
import torch
from collections import namedtuple
import sys

sys.path.append("..")
from deep_rl.ppo import PPO
from sabre360_with_qoe import SessionInfo, SessionEvents
from utils import get_tiles_in_viewport, Pose2VideoXY
from abr.TiledAbr import TiledAbr

''' 360-degree video设置 '''
TILES_X = 8
TILES_Y = 8
BITRATE_LEVEL = 6

''' 算法超参数设置 '''
SLEEP_PERIOD = 500  # 暂停时长 ms
TPUT_HISTORY_LEN = 10  # 历史吞吐量
MAX_ET_LEN = 5  # 最大的ET buffer长度 s

''' PPO设置 '''
STATE_DIMENSION = 36
HISTORY_LENGTH = 1
ACTION_DIMENSION = 27
NN_MODEL = "deep_rl/model/PPO_et_update_0_500.pth"
lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network
K_epochs = 80  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.85  # discount factor
device = torch.device('cpu')

TiledAction = namedtuple('TiledAction', 'segment tile quality delay')


def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)


TiledAction.__str__ = str_tiled_action


class Melody(TiledAbr):
    def __init__(self, config, session_info: SessionInfo, session_events: SessionEvents):
        # print("<< proposed rl-based abr >>")
        self.session_info = session_info
        self.session_events = session_events
        self.manifest = self.session_info.get_manifest()
        self.buffer = self.session_info.get_buffer()
        self.video_size = self.manifest.segments  # [segment][tile][level]
        self.video_time = self.manifest.segment_duration * len(self.video_size)
        self.bitrate_level = self.manifest.bitrates

        self.last_action = None
        self.first_step = True
        self.is_sleep = False
        self.is_BT_download = False
        self.is_ET_download = False
        # ============ 初始化Agent ============
        self.ppo_agent = PPO(STATE_DIMENSION, ACTION_DIMENSION, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
        self.ppo_agent.load(NN_MODEL)
        # ============ 初始化State及相关变量 ============
        self.state = torch.zeros(HISTORY_LENGTH, STATE_DIMENSION)
        self.past_k_tput = []
        self.avg_level_dict = {}
        self.base_buffer_depth = 0
        self.enhance_buffer_depth = 0
        self.latest_BT_segment = -1
        self.latest_ET_segment = -1
        self.pred_view_dict = {}
        self.pred_tiles_dict = {}
        self.contents = {}

        self.acc = np.zeros(MAX_ET_LEN)
        self.acc_count = 0

    def initialize(self):
        ''' 初始化状态，返回state'''
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
        future_size_in_BT, future_size_in_ET, tile_num_in_ET = self.update_download_size(first_segment)
        s_future_size = future_size_in_BT + future_size_in_ET
        assert len(s_future_size) == 1 + MAX_ET_LEN
        s_future_size = np.array(s_future_size) / 1000000.
        self.state[0, 12:18] = torch.Tensor(s_future_size)
        tile_num_in_ET = np.array(tile_num_in_ET) / 10.
        self.state[0, 18:23] = torch.Tensor(tile_num_in_ET)

        # 平均码率等级
        start_et_segment = 0
        end_et_segment = start_et_segment + MAX_ET_LEN
        s_avg_level = []
        for segment_idx in range(start_et_segment, end_et_segment + 1):
            s_avg_level.append(0)
        assert len(s_avg_level) == MAX_ET_LEN + 1
        self.state[0, 18:24] = torch.Tensor(s_avg_level)

        # 视点运动信息
        std_x, std_y = self.calculate_pose_spead(first_segment, 0)
        self.state[0, 24] = std_x * 100
        self.state[0, 25] = std_y * 100

        # 视角预测准确度
        self.acc = self.calculate_pred_acc()
        self.state[0, 26:31] = torch.Tensor(self.acc)

    def get_action(self):
        ''' 决策 '''
        if self.first_step:
            self.initialize()  # 初始化state
            ppo_output = self.ppo_agent.select_action(self.state)
            self.first_step = False
        else:
            self.calculate_state()  # 更新state
            ppo_output = self.ppo_agent.select_action(self.state)
            # print(ppo_output)
        action = self.transform_action(ppo_output)
        self.last_action = action
        # print('')
        # print('bt buffer length= {}'.format(self.base_buffer_depth))
        # print('et buffer length= {}'.format(self.enhance_buffer_depth))
        # print(action, '\n')
        return action

    def calculate_state(self):
        # ================== 变量更新 ==================
        playhead = self.buffer.get_play_head()
        first_segment = self.buffer.get_played_segments()
        first_segment_offset = self.buffer.get_played_segment_partial()
        first_et_segment = first_segment
        if first_segment_offset != 0:
            first_et_segment += 1
        segment_depth = self.buffer.get_buffer_depth()
        self.contents = {}  # 每个segment内的tile信息，未被下载的是None
        for i in range(segment_depth):
            segment_id = first_segment + i
            self.contents[segment_id] = self.buffer.get_buffer_contents(segment_id)

        if not self.is_sleep:
            self.update_throughput(self.last_action)
        self.update_buffer_length(self.last_action, self.is_BT_download, first_et_segment)
        self.update_viewport_pred(first_segment, first_et_segment)
        future_size_in_BT, future_size_in_ET, tile_num_in_ET = self.update_download_size(first_et_segment)
        self.update_avg_level(first_segment, first_et_segment, self.contents)

        # ================== state计算 ==================
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

        # 待下载的数据量
        s_future_size = future_size_in_BT + future_size_in_ET
        assert len(s_future_size) == 1 + MAX_ET_LEN
        s_future_size = np.array(s_future_size) / 1000000.
        self.state[0, 12:18] = torch.Tensor(s_future_size)
        tile_num_in_ET = np.array(tile_num_in_ET) / 10
        self.state[0, 18:23] = torch.Tensor(tile_num_in_ET)

        # 平均码率等级
        end_et_segment = first_et_segment + MAX_ET_LEN
        s_avg_level = []
        for segment_idx in range(first_et_segment, end_et_segment + 1):
            if segment_idx in self.avg_level_dict:
                s_avg_level.append(self.avg_level_dict[segment_idx])
            else:
                s_avg_level.append(0)
        assert len(s_avg_level) == MAX_ET_LEN + 1
        self.state[0, 18:24] = torch.Tensor(s_avg_level)

        # 视点运动信息
        std_x, std_y = self.calculate_pose_spead(first_segment, first_segment_offset)
        self.state[0, 24] = std_x * 100
        self.state[0, 25] = std_y * 100

        # 视角预测准确度(5个step更新一次)
        if self.acc_count % MAX_ET_LEN == 0:
            self.acc = self.calculate_pred_acc()
            self.acc_count += 1
        self.state[0, 31:36] = torch.Tensor(self.acc)

    def transform_action(self, ppo_output):
        action = []
        self.is_sleep = False
        self.is_BT_download = False
        self.is_ET_download = False
        if ppo_output == ACTION_DIMENSION - 1:
            ''' delay '''
            action.append(TiledAction(-1, -1, -1, SLEEP_PERIOD))
            self.is_sleep = True
        elif ppo_output == ACTION_DIMENSION - 2:
            ''' bt '''
            self.is_BT_download = True
            segment_id = self.latest_BT_segment + 1
            assert segment_id <= len(self.manifest.segments) - 1
            for tile_id in range(TILES_X * TILES_Y):
                action.append(TiledAction(segment_id, tile_id, 0, 0))
        else:
            ''' et '''
            self.is_ET_download = True
            start_idx = self.buffer.get_played_segments()
            offset = self.buffer.get_played_segment_partial()
            if offset != 0:
                start_idx += 1
            segment_id = int(start_idx + ppo_output // (BITRATE_LEVEL - 1))
            assert segment_id < self.latest_BT_segment + 1
            assert segment_id <= len(self.manifest.segments) - 1
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
        return action

    def update_throughput(self, action):
        ''' 更新历史带宽信息 '''
        size = 0
        for i in range(len(action)):
            size += self.video_size[action[i].segment][action[i].tile][action[i].quality]  # bit
        download_time = self.session_info.get_total_download_time()  # ms
        assert download_time != 0
        tput = size / download_time  # kbps
        if len(self.past_k_tput) >= TPUT_HISTORY_LEN:
            self.past_k_tput.pop(0)
        self.past_k_tput.append(tput)

    def update_buffer_length(self, action, is_BT_download, first_et_segment):
        ''' 更新ET和BT buffer的长度 '''
        is_BT_download_re = False  # todo：可以删掉
        ''' 更新ET和BT buffer的长度 '''
        # 1. 更新bt长度
        if action[0].segment > self.latest_BT_segment:
            is_BT_download_re = True
        assert is_BT_download_re == is_BT_download
        self.latest_BT_segment = max(action[0].segment, self.latest_BT_segment)
        buffer_depth = self.buffer.get_buffer_depth()
        offset = self.buffer.get_played_segment_partial() / 1000.
        self.base_buffer_depth = buffer_depth - offset  # 基础层长度即缓冲区实际长度

        # 2. 更新et更新过的视频个数
        self.enhance_buffer_depth = 0
        for i in range(MAX_ET_LEN):
            segment_id = first_et_segment + i
            contents = self.buffer.get_buffer_contents(segment_id)
            if contents is None:
                continue
            for tile_level in contents:
                if tile_level is not None and tile_level > 0:
                    self.enhance_buffer_depth += 1
                    break
        assert self.enhance_buffer_depth <= MAX_ET_LEN

    def update_viewport_pred(self, first_segment, first_et_segment):
        ''' 更新视角预测结果 (对buffer内所有tile) '''
        # 1. 执行一次视角预测
        start_segment_idx = first_et_segment  # 确定ET层可下载的范围 start~end（当前播放视频也预测,end是最后一个可下载的segment）
        end_segment_idx = start_segment_idx + MAX_ET_LEN
        pred_view_dict = {}
        model_x, model_y = self.session_info.get_viewport_predictor().build_model(end_segment_idx)
        if model_x is None:
            for seg_idx in range(first_segment, end_segment_idx + 1):
                pred_view_dict[seg_idx] = [0.5, 0.5]  # 默认视点 (0.5,0.5)
        else:
            for seg_idx in range(first_segment, end_segment_idx + 1):
                pred_view = self.session_info.get_viewport_predictor().predict_view(model_x, model_y, seg_idx)
                (x_pred, y_pred) = pred_view
                pred_view_dict[seg_idx] = [x_pred, y_pred]
        # 2 根据预测结果确定视窗内的tile
        self.pred_tiles_dict = {}
        for seg_idx in range(first_segment, end_segment_idx + 1):
            x_pred = pred_view_dict[seg_idx][0]
            y_pred = pred_view_dict[seg_idx][0]
            tiles_in_viewport = get_tiles_in_viewport(x_pred, y_pred)
            self.pred_tiles_dict[seg_idx] = tiles_in_viewport

    def update_download_size(self, first_et_segment):
        ''' 更新待下载的数据量 '''
        # ======> Part 1: segment in BT buffer - 最低质量全画幅下载
        size_in_BT = []
        next_BT_segment = self.latest_BT_segment + 1
        download_bit = 0
        if next_BT_segment <= len(self.video_size) - 1:  # 有未下载的BT层segment
            for i in range(len(self.video_size[next_BT_segment])):
                download_bit += self.video_size[next_BT_segment][i][0]
        else:  # 没有未下载的BT层segment
            download_bit = 0
        size_in_BT.append(download_bit)

        # ======> Part 2: segments in ET buffer - 待更新码率
        size_in_ET, num_in_ET = [], []
        for i in range(MAX_ET_LEN):  # 遍历et buffer中的segment
            download_bit = 0
            segment_idx = first_et_segment + i
            pred_tiles = self.pred_tiles_dict[segment_idx]
            if segment_idx >= len(self.video_size) - 1:  # 超过segment idx
                size_in_ET.append(download_bit)
                num_in_ET.append(0)
                continue

            content = self.buffer.get_buffer_contents(segment_idx)
            if content is None:  # bt层没下载过
                size_in_ET.append(0)
                num_in_ET.append(0)
                continue

            cnt = 0
            for tile_id in pred_tiles:
                assert content[tile_id] is not None
                if content[tile_id] == 0:
                    cnt += 1
                    download_bit += self.video_size[segment_idx][tile_id][-1]
            size_in_ET.append(download_bit)
            num_in_ET.append(cnt)

        return size_in_BT, size_in_ET, num_in_ET  # bits

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
        pose_trace = self.session_info.get_prev_pose_trace()
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

    def calculate_pred_acc(self):
        acc = [0.75, 0.68, 0.62, 0.60, 0.57]
        pose_trace = self.session_info.get_prev_pose_trace()
        if len(pose_trace) == 0:
            return acc
        latest_segment = list(pose_trace.keys())[-1]
        latest_pose = pose_trace.get(list(pose_trace.keys())[-1])
        real_pose = Pose2VideoXY(latest_pose)
        view_predictor = self.session_info.get_viewport_predictor()
        for i in range(MAX_ET_LEN):
            pred_len = i + 1
            model_x, model_y = view_predictor.build_model_to_get_acc(pred_len, latest_segment)
            if model_x is None:
                continue
            pred_pose = view_predictor.predict_view(model_x, model_y, latest_segment)
            pred_acc = view_predictor.get_pred_acc(pred_pose, real_pose)
            acc[i] = pred_acc
        return np.array(acc)
