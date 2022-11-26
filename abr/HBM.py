''' ###########################
    #   SJTU's two-tier ABR   #
    #   有问题，不用作baseline  #
    ###########################'''

import math
from collections import namedtuple
import numpy as np
from abr.TiledAbr import TiledAbr
from utils import get_tiles_in_viewport, calculate_viewing_proportion
from sabre360_with_qoe import SessionInfo, SessionEvents

TiledAction = namedtuple('TiledAction', 'segment tile quality delay')
TILES_X = 8
TILES_Y = 8

# 算法参数
MAX_BUFFER_LEVEL = 5  # s
BUFFER_THRESHOLD = 2  # s
TPUT_HISTORY_LEN = 10  # s
ALPHA = 0.9


def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)


TiledAction.__str__ = str_tiled_action


class HBM(TiledAbr):
    def __init__(self, config, session_info: SessionInfo, session_events: SessionEvents):
        self.session_info = session_info
        self.session_events = session_events

        self.manifest = self.session_info.get_manifest()
        self.bitrate = self.manifest.bitrates
        self.video_size = self.manifest.segments
        self.segment_duration = self.manifest.segment_duration
        self.max_segment_id = len(self.video_size) - 1

        self.buffer = self.session_info.get_buffer()

        self.view_predictor = self.session_info.get_viewport_predictor()
        self.model_x, self.model_y = None, None

        # 算法变量
        self.past_tput = []
        self.latest_bt_segment = -1
        self.bwe = 0

        self.last_action = None

    def get_action(self):
        ''' HBM主要逻辑 '''

        # 带宽估计
        tput = self.get_throughput()
        if tput is not None:
            if len(self.past_tput) >= TPUT_HISTORY_LEN:
                self.past_tput.pop(0)
            self.past_tput.append(tput)
        self.bwe = self.bandwidth_estimation()
        buffer_length = self.buffer.get_buffer_depth() - self.buffer.get_played_segment_partial() / 1000.
        request_rate = ALPHA ** (MAX_BUFFER_LEVEL - buffer_length) * self.bwe  # 确定请求码率 kbps
        request_rate *= 1024.  # bps

        # 缓冲区状态更新
        play_head = self.buffer.get_play_head() / 1000.  # s

        # 码率选择
        # -------------- 限制条件 --------------
        # 1. 总码率低于请求码率
        # 2. 相邻tile码率等级不超过两个级别
        max_utility = float('-inf')
        et_action = None
        bt_segment = self.latest_bt_segment + 1
        download_bt_flag = False
        if bt_segment <= self.max_segment_id and bt_segment <= play_head + MAX_BUFFER_LEVEL:
            download_bt_flag = True
        action = None
        for bt_level in range(len(self.bitrate)):
            total_utility = 0
            ''' 1. 对vd的每一种码率状态都做循环 '''
            bt_rate = 0
            if download_bt_flag:  # 能下载bt
                # 有et层可下载且不超过最大缓冲区大小
                for tile_id in range(TILES_X * TILES_Y):
                    bt_rate += self.video_size[bt_segment][tile_id][bt_level]
                    w_space = self.get_space_weight(bt_segment, tile_id)
                    total_utility += w_space * self.bitrate[bt_level]
            available_rate = request_rate - bt_rate  # bps
            if available_rate < 0:  # 不足以支持bt的下载
                if bt_level > 0:
                    break
                else:
                    for tile_id in range(TILES_X * TILES_Y):
                        action.append(TiledAction(bt_segment, tile_id, 0, 0))
            ''' 2. 开始遍历更新B_th前的segment '''
            first_segment = self.buffer.get_played_segments()
            if self.buffer.get_played_segment_partial() > 0:
                first_segment += 1
            update_length = min(self.latest_bt_segment - first_segment + 1, BUFFER_THRESHOLD)

            # 视角预测
            self.model_x, self.model_y = self.view_predictor.build_model(first_segment + update_length - 1)
            segment_utility = []  # [segment][tile][bitrate]=utility
            for segment in range(update_length):  # todo: 下载选项更多
                tiles_utility = []
                for tile in range(TILES_X * TILES_Y):  # 遍历每个tile计算utility
                    level_utility = []
                    for bitrate in range(len(self.bitrate)):
                        w_space = self.get_space_weight(segment, tile)
                        utility = self.get_utility_over_cost(w_space, segment, tile, bitrate)
                        level_utility.append(utility)
                    tiles_utility.append(level_utility[:])
                segment_utility.append(tiles_utility[:])

            # 贪心算法
            utility_array = np.array(segment_utility)
            used_rate = 0
            download_time = 0
            update_action = []
            while used_rate < available_rate:  # 贪心遍历
                utility = np.max(utility_array)
                if utility <= 0:
                    break
                segment, tile, bitrate = np.where(utility_array == utility)
                # ================= 限制条件 =================
                w_time = self.get_time_weight(download_time, segment, tile, bitrate)
                if w_time <= 0:  # 来不及下载，不下载
                    utility_array[segment, tile, bitrate] = -1
                    continue
                if not self.check_smooth(segment, tile, bitrate):  # 限制码率变化
                    utility_array[segment, tile, bitrate] = -1
                    continue

                size = self.video_size[segment][tile][bitrate]
                download_time += self.bwe / size  # ms
                total_utility += utility
                # 加入下载列表
                update_action.append(TiledAction(segment, tile, bitrate, 0))

            if total_utility > max_utility:
                max_utility = total_utility
                et_action = update_action
                bt_action = []
                if download_bt_flag:
                    for tile in range(TILES_X * TILES_Y):
                        bt_action.append(TiledAction(bt_segment, tile, bt_level, 0))
                    action = bt_action.append(et_action)
                else:
                    action = et_action

            if not download_bt_flag:  # 不下载bt,不做后续遍历
                break
        self.last_action = action

        return action

    def get_throughput(self):
        ''' 带宽估计模块 '''
        if self.last_action is None:
            return 10000  # kbps
        if self.last_action[0].delay > 0:  # sleep
            return None
        download_time = self.session_info.get_total_download_time()  # ms
        assert download_time > 0
        size = 0.0
        for i in range(len(self.last_action)):
            segment_id = self.last_action[i].segment
            tile_id = self.last_action[i].tile
            bitrate_level = self.last_action[i].quality
            size += self.video_size[segment_id][tile_id][bitrate_level]
        tput = size / download_time
        return tput

    def bandwidth_estimation(self):
        ''' 带宽估计模块(调和平均) '''
        sum = 0
        for i in range(len(self.past_tput)):
            sum += 1 / self.past_tput[i]
        bwe = len(self.past_tput) / sum
        return bwe

    def get_utility_over_cost(self, weight, segment, tile, bitrate):
        ''' 计算效用代价函数 '''
        rate = self.video_size[segment][tile][bitrate] / 1024.  # kbps
        cur_bitrate = self.buffer.get_buffer_element(segment, tile)
        if cur_bitrate is None:
            cur_bitrate = 0
        else:
            cur_bitrate = self.bitrate[cur_bitrate]  # kbps
        new_bitrate = self.bitrate[bitrate]

        utility = weight * (new_bitrate - cur_bitrate) / rate
        return utility

    def get_space_weight(self, segment, tile):
        ''' 计算空域权重 '''
        x_pred, y_pred = self.view_predictor.predict_view(self.model_x, self.model_y, segment)
        proportion = calculate_viewing_proportion(x_pred, y_pred, tile)
        sum_proportion = 0
        for i in range(TILES_X * TILES_Y):
            sum_proportion += calculate_viewing_proportion(self.model_x, self.model_y, i)
        return proportion / sum_proportion

    def get_time_weight(self, prev_download_time, segment, tile, bitrate):
        ''' 计算时间权重 '''
        play_deadline = segment * self.segment_duration - self.buffer.get_play_head()  # ms
        size = self.video_size[segment][tile][bitrate]  # bps
        download_time = prev_download_time + self.bwe / size  # ms
        w_time = 1.0 / (play_deadline - download_time)
        return w_time

    def check_smooth(self, segment, tile, download_bit_level):
        ''' 计算平滑度的限制条件 '''
        #  周围tile已有码率
        buffered_bit_level = []
        delta_x = [-1, 1]
        delta_y = [-1, 1]
        pos_x = tile % TILES_X
        pos_y = tile // TILES_X
        for i in delta_x:
            for j in delta_y:
                buffered_x = pos_x + delta_x[i]
                buffered_x = buffered_x % TILES_X

                buffered_y = pos_y + delta_y[j]
                if buffered_y < 0 or buffered_y >= TILES_Y:
                    continue
                buffered_tile = buffered_y * TILES_X + buffered_x
                level = self.buffer.get_buffer_element(segment, buffered_tile)
                assert level is not None
                buffered_bit_level.append(level)
        max_level = max(buffered_bit_level)
        min_level = min(buffered_bit_level)

        if abs(max_level - download_bit_level) > 2 or abs(min_level - buffered_bit_level) > 2:
            return False
        else:
            return True
