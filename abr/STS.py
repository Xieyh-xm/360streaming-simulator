import math
from collections import namedtuple
import numpy as np
from utils import get_trace_file, get_tiles_in_viewport, calculate_viewing_proportion, Pose2VideoXY
from abr.TiledAbr import TiledAbr
from sabre360_with_qoe import SessionInfo, SessionEvents
from myLog import myLog

TiledAction = namedtuple('TiledAction', 'segment tile quality delay')
AvailableAction = namedtuple('AvailableAction', 'segment download_tile_dict')

TILES_X = 8
TILES_Y = 8

TPUT_WINDOWS = 10
SLEEP_PERIOD = 100

LOG_PATH = "./log/RAM360.log"
LOG_FLAG = False


def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)


TiledAction.__str__ = str_tiled_action


class STS(TiledAbr):
    def __init__(self, config, session_info: SessionInfo, session_events: SessionEvents):
        self.session_info = session_info
        self.session_events = session_events
        self.max_depth = math.floor(session_info.buffer_size / session_info.get_manifest().segment_duration)
        self.max_segment_id = len(session_info.get_manifest().segments) - 1

        self.manifest = self.session_info.get_manifest()
        self.bitrate = self.manifest.bitrates
        self.video_size = self.manifest.segments
        self.segment_duration = self.manifest.segment_duration
        self.buffer = self.session_info.get_buffer()

        self.view_predictor = self.session_info.get_viewport_predictor()
        self.pred_view = {}

        self.past_tput = []
        self.latest_seg = -1
        self.last_action = None
        self.sleep_flag = False

        self.log = myLog(LOG_PATH)
        self.bwe = 10000.

    def get_action(self):

        # 带宽估计
        if not self.sleep_flag:  # 没有sleep
            tput = self.calculate_throughput(self.last_action)
            if len(self.past_tput) >= TPUT_WINDOWS:
                self.past_tput.pop(0)
            self.past_tput.append(tput)
            self.bwe = self.bandwidth_estimation()

        cur_buffer_length = self.buffer.get_buffer_depth() * 1000 - self.buffer.get_played_segment_partial()  # ms
        if cur_buffer_length >= 3000.:
            action = [TiledAction(0, 0, 0, SLEEP_PERIOD)]
            self.sleep_flag = True
            return action

        # 全部chunk都下载完成
        if self.latest_seg + 1 <= self.max_segment_id:
            self.latest_seg += 1
        else:
            action = [TiledAction(0, 0, 0, SLEEP_PERIOD)]
            self.sleep_flag = True
            return action

        # 视角预测
        model_x, model_y = self.view_predictor.build_model(self.latest_seg)
        if model_x is None:
            pred_view = (0.5, 0.5)
        else:
            pred_view = self.view_predictor.predict_view(model_x, model_y, self.latest_seg)
        (x_pred, y_pred) = pred_view
        tiles_in_viewport = get_tiles_in_viewport(x_pred, y_pred)

        # 码率分配
        download_time = []
        for bit_level in range(len(self.bitrate)):
            size = 0
            for tile in range(TILES_X * TILES_Y):
                if tile in tiles_in_viewport:
                    size += self.video_size[self.latest_seg][tile][bit_level]
                else:
                    size += self.video_size[self.latest_seg][tile][0]
            time = size / self.bwe  # ms
            download_time.append(time)

        best_choice = 0
        min_delta = float('inf')
        for level, time in enumerate(download_time):
            if abs((cur_buffer_length - time) - 2000.) < min_delta:
                best_choice = level
                min_delta = abs((cur_buffer_length - time) - 2000.)
        # print(best_choice)

        action = []
        for i in range(TILES_X * TILES_Y):
            if i in tiles_in_viewport:
                action.append(TiledAction(self.latest_seg, i, best_choice, 0))
            else:
                action.append(TiledAction(self.latest_seg, i, 0, 0))
        self.sleep_flag = False
        self.last_action = action
        return action

    def calculate_throughput(self, action):
        ''' 返回最新吞吐量 '''
        if action is None:  # 起始阶段默认吞吐量
            return 10000
        size = 0
        for i in range(len(action)):
            size += self.video_size[action[i].segment][action[i].tile][action[i].quality]  # bit
        download_time = self.session_info.get_total_download_time()  # ms
        assert download_time != 0
        tput = size / download_time  # kbps
        return tput

    def bandwidth_estimation(self):
        sum = 0
        for i in range(len(self.past_tput)):
            sum += 1 / self.past_tput[i]
        return len(self.past_tput) / sum  # kbps
