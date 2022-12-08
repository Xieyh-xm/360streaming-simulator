''' ###########################
    # Wang Yao's two-tier ABR #
    ###########################'''
import math
from collections import namedtuple
import numpy as np
from abr.TiledAbr import TiledAbr
from utils import get_tiles_in_viewport
from sabre360_with_qoe import SessionInfo, SessionEvents
from myLog import myLog

TiledAction = namedtuple('TiledAction', 'segment tile quality delay')
TILES_X = 8
TILES_Y = 8

alpha = 0
gamma = 0
target_bt_length = 10
target_et_length = 1  # todo: candidate 1~4s
PROPORTIONAL_GAIN = 0.6
INTEGRATION_GAIN = 0.01
HISTORY_LENGTH = 10  # s

SLEEP_PERIOD = 500  # 暂停时长 ms

LOG_FLAG = False
LOG_PATH = './log/TTS.log'


def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)


TiledAction.__str__ = str_tiled_action


class TTS(TiledAbr):
    def __init__(self, config, session_info: SessionInfo, session_events: SessionEvents):
        self.session_info = session_info
        self.session_events = session_events

        self.manifest = self.session_info.get_manifest()
        self.bitrate = self.manifest.bitrates
        self.video_size = self.manifest.segments
        self.segment_duration = self.manifest.segment_duration
        self.buffer = self.session_info.get_buffer()

        self.view_predictor = self.session_info.get_viewport_predictor()

        # 缓冲区状态
        self.play_head = 0
        self.latest_et_segment = -1
        self.latest_bt_segment = -1
        self.et_buffer_length = 0
        self.bt_buffer_length = 0

        self.play_time = []
        self.history_et_length = []

        self.last_action = None
        self.log = myLog(LOG_PATH)

    def get_action(self):
        ''' TTS主逻辑'''
        is_sleep = False

        # 更新缓冲区大小
        self.play_head = self.buffer.get_play_head() / 1000.  # s
        self.bt_buffer_length = self.buffer.get_buffer_depth() - self.buffer.get_played_segment_partial() / 1000.
        self.et_buffer_length = max(self.latest_et_segment - self.play_head, 0)
        if LOG_FLAG:
            self.log.log_playhead(self.play_head*1000.)
            self.log.logger.info(
                "bt_buffer_length : {} \tet_buffer_length : {}".format(self.bt_buffer_length, self.et_buffer_length))

        self.play_time.append(self.play_head)
        self.history_et_length.append(self.et_buffer_length)

        if self.bt_buffer_length <= target_bt_length and self.latest_bt_segment + 1 < len(self.video_size):
            ''' 下载全画幅bt层 '''
            is_bt_download = True
            segment_id = self.latest_bt_segment + 1
            # 1. 确定action
            action = []
            for i in range(TILES_X * TILES_Y):  # 低质量全画幅
                action.append(TiledAction(segment_id, i, 0, 0))
            # 2. 更新相关变量
            self.latest_bt_segment += 1
            if LOG_FLAG:
                self.log.log_bt_action(segment_id)
        elif self.et_buffer_length <= target_et_length:
            ''' 下载et层 '''
            # 确定et segment
            first_et_segment = self.buffer.get_played_segments()
            if self.buffer.get_played_segment_partial() > 0:
                first_et_segment += 1
            segment_id = max(self.latest_et_segment + 1, first_et_segment)
            if segment_id >= len(self.video_size):
                ''' 没有et segment可下载，系统空闲一段时间 '''
                is_sleep = True
                action = [TiledAction(0, 0, 0, SLEEP_PERIOD)]
                if LOG_FLAG:
                    self.log.log_sleep(SLEEP_PERIOD)
            else:
                is_et_download = True
                # 1. 估计未来带宽bwe
                bwe = self.calculate_throughput()  # bps
                # 2. 视角预测
                model_x, model_y = self.view_predictor.build_model(segment_id)
                if model_x is None:
                    pred_view = (0.5, 0.5)
                else:
                    pred_view = self.view_predictor.predict_view(model_x, model_y, segment_id)
                (x_pred, y_pred) = pred_view
                tiles_in_viewport = get_tiles_in_viewport(x_pred, y_pred)
                # 3. 选择码率等级
                integration = self.sum_past_delta()  # 计算过去10s的累加
                # print("integration = ", integration)
                assert integration <= 10.
                control_signal = PROPORTIONAL_GAIN * (
                        self.et_buffer_length - target_et_length) + INTEGRATION_GAIN * integration
                et_rate = min(control_signal + 1, segment_id - self.play_head) * bwe  # kbps
                # 4. 码率分配
                et_rate *= 1024.  # bps
                # print("control_signal = ", control_signal)
                # print("self.et_buffer_length - target_et_length = ", self.et_buffer_length - target_et_length)
                # print(et_rate)
                choose_level = 1
                for bitrate_level in range(1, len(self.manifest.bitrates)):
                    overflow = False
                    size = 0
                    for tile_id in tiles_in_viewport:
                        size += self.video_size[segment_id][tile_id][bitrate_level]
                        if size > et_rate:
                            overflow = True
                            break
                    if not overflow:
                        choose_level = bitrate_level
                    else:
                        break
                # 5. 确定action
                action = []
                # print("choose_level = ", choose_level)
                for tile_id in tiles_in_viewport:
                    action.append(TiledAction(segment_id, tile_id, choose_level, 0))
                # 6. 更新相关变量
                self.latest_et_segment = segment_id
                if LOG_FLAG:
                    self.log.log_et_action(segment_id, choose_level)
        else:
            ''' 系统空闲一段时间 '''
            is_sleep = True
            action = [TiledAction(0, 0, 0, SLEEP_PERIOD)]
            if LOG_FLAG:
                self.log.log_sleep(SLEEP_PERIOD)

        if not is_sleep:
            self.last_action = action
        return action

    def calculate_throughput(self):
        ''' 计算最新吞吐量 '''
        if self.last_action is None:
            return 10000  # kbps
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

    def sum_past_delta(self):
        ''' 累加过去10s et层缓冲区的波动 '''
        start_time = max(0, self.play_head - HISTORY_LENGTH)
        integration = 0
        # for i in range(min(HISTORY_LENGTH, int(self.play_time[-1]))):
        #     index = 0
        #     while start_time > self.play_time[index]:
        #         index += 1
        #     et_length = self.history_et_length[index] - (start_time - self.play_time[index])
        #     delta = et_length - target_et_length
        #     integration += delta
        #     start_time += 1
        for i in range(min(HISTORY_LENGTH, len(self.play_time))):
            et_length = self.history_et_length[-i]
            delta = et_length - target_et_length
            integration += delta
        # print(integration)
        return integration
