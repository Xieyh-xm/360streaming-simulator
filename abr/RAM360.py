import math
from collections import namedtuple
import numpy as np
from utils import get_trace_file, get_tiles_in_viewport, calculate_viewing_proportion

TiledAction = namedtuple('TiledAction', 'segment tile quality delay')
AvailableAction = namedtuple('AvailableAction', 'segment download_tile_dict')

LAMDA = 0.1
CONTROL_PARM_V = 0.28
STALL_TH = 3.0
BANDWIDTH_TH = 0.5
TILES_X = 8
TILES_Y = 8
DEFAULT_PRED_ACC = [0.75, 0.68, 0.62, 0.60, 0.57, 0.55, 0.52, 0.51, 0.49, 0.48]


def str_tiled_action(self):
    seconds = 0 if self.delay is None else (self.delay / 1000)
    return 'segment:%d tile:%d quality:%d delay:%.3fs' % (self.segment, self.tile, self.quality, seconds)


TiledAction.__str__ = str_tiled_action


class TiledAbr:

    # TODO: rewrite report_*() to use SessionEvents

    def __init__(self):
        pass

    def get_action(self):
        raise NotImplementedError

    def check_abandon(self, progress):
        return None

    def report_action_complete(self, progress):
        pass

    def report_action_cancelled(self, progress):
        pass

    def report_seek(self, where):
        raise NotImplementedError


class RAM360(TiledAbr):
    def __init__(self, config, session_info, session_events):
        self.session_info = session_info
        self.session_events = session_events
        self.max_depth = math.floor(session_info.buffer_size / session_info.get_manifest().segment_duration)
        self.max_segment_id = len(session_info.get_manifest().segments) - 1

        self.qoe_estimator = QoeEstimator(TILES_X, TILES_Y, self.session_info.get_buffer())
        self.manifest = self.session_info.get_manifest()
        self.segment_duration = self.manifest.segment_duration
        self.buffer = self.session_info.get_buffer()
        self.pred_acc = {}
        for i in range(len(DEFAULT_PRED_ACC)):
            self.pred_acc[i + 1] = DEFAULT_PRED_ACC[i]


def get_action(self):
    # 变量更新
    first_segment = self.buffer.get_played_segments()
    first_segment_offset = self.buffer.get_played_segment_partial()

    first_unplayed_segment = first_segment  # l_{k}
    unplayed_segment_num = self.buffer.get_buffer_depth()  # L_{k}
    if first_segment_offset != 0:
        first_unplayed_segment = min(first_unplayed_segment + 1, self.max_segment_id)
        unplayed_segment_num -= 1
    max_download_segment_id = max(first_unplayed_segment + unplayed_segment_num, self.max_segment_id)

    # 遍历可下载的每一个segment
    max_utility = 0
    optimal_action = None
    for segment_id in range(first_unplayed_segment, max_download_segment_id + 1):
        '''============= 视点预测 ============='''
        model_x, model_y = self.session_info.get_viewport_predictor().build_model(segment_id)
        pred_view = self.session_info.get_viewport_predictor().predict_view(model_x, model_y, segment_id)
        if pred_view is not None:
            (x_pred, y_pred) = pred_view
        else:
            x_pred, y_pred = 0.5, 0.5
        tiles_in_viewport = get_tiles_in_viewport(x_pred, y_pred)  # 视窗内的tile

        utility = 0
        cur_action = None
        if segment_id < max_download_segment_id:
            '''============= update ============='''
            # todo:码率选择的循环
            pass
        elif segment_id == max_download_segment_id:
            '''============= prefetch ============='''
            download_bit = 0  # b_{k}
            download_tile = {}
            for i in range(TILES_X * TILES_Y):  # 全画幅下载
                download_bit += self.manifest.segment[segment_id][i][0]
                download_tile[i] = 0
            buffer_length = self.buffer.get_buffer_depth() - first_segment_offset / 1000.  # Q_{k} ms
            # todo: I_{k}的计算
            cur_action = AvailableAction(segment=segment_id, download_tile_dict=download_tile)
            quality_improvement = 0
            # todo: pred_acc计算---不同窗口预测当前播放的segment，写在view_prediction文件里


            utility = (CONTROL_PARM_V * quality_improvement - buffer_length * self.segment_duration) / download_bit

        if utility > max_utility:
            max_utility = utility
            optimal_action = cur_action

    # 将 AvailableAction 转换为 TiledAction
    optimal_segment = optimal_action.segment
    optimal_download_tile = optimal_action.download_tile_dict
    action = []
    for tile_id in optimal_download_tile:
        action.append(TiledAction(optimal_segment, tile_id, optimal_download_tile[tile_id], 0))
    return action


class QoeEstimator:
    def __init__(self, tiles_x, tiles_y, buffer):
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.tiles_num = tiles_x * tiles_y
        self.buffer = buffer

    def get_oscillation_space(self, avg_quality, quality_list_non, video_xy):
        ''' 计算segment空间平滑度(非差值) '''
        sum_delta = 0
        total_proportion = 0
        (video_x, video_y) = video_xy
        for tile_id in range(len(quality_list_non)):
            proportion = calculate_viewing_proportion(video_x, video_y, tile_id)
            total_proportion += proportion
            sum_delta += proportion * (quality_list_non[tile_id] - avg_quality) ** 2
        std_space = math.sqrt(sum_delta / total_proportion)
        return std_space

    def get_quality(self, segment, download_tiles: dict, video_xy, pred_acc):
        ''' 计算segment质量(非差值) '''
        (video_x, video_y) = video_xy
        is_download = False
        quality_list = []
        quality_non_list = []
        total_proportion = 0
        for tile_id in range(self.tiles_num):
            # 1. 获取tile质量
            buffered_quality = self.buffer.get_buffer_element(segment, tile_id)  # buffer中已缓存的tile质量
            if buffered_quality is None:
                buffered_quality = 0
            download_quality = 0  # 待下载的tile质量
            if tile_id in download_tiles:
                download_quality = download_tiles[tile_id]
                is_download = True

            # 2. 计算tile proportion
            proportion = calculate_viewing_proportion(video_x, video_y, tile_id)
            total_proportion += proportion

            # 3. 计算质量
            if is_download:
                quality = proportion * (buffered_quality + pred_acc * (download_quality - buffered_quality))
                quality_list.append(quality)

                quality_non = buffered_quality + pred_acc * (download_quality - buffered_quality)
                quality_non_list.append(quality_non)
            else:
                quality = proportion * buffered_quality
                quality_list.append(quality)

                quality_non = buffered_quality
                quality_non_list.append(quality_non)

        quality_list = np.array(quality_list)
        quality_list = quality_list / total_proportion
        return quality_list

    def get_total_quality(self, segment, download_tiles, pred_view, pred_acc):
        ''' 计算一个segment总平均质量 '''
        # 1. 计算segment质量(非差值)
        video_xy = pred_view[segment]
        quality_list, quality_list_non = self.get_quality(segment, download_tiles, video_xy,
                                                          pred_acc)  # 加权的quality和未加权的quality
        avg_quality = np.sum(quality_list)
        oscillation_time, oscillation_space = 0, 0
        # 2. 计算时间平滑度
        if segment != 0:
            video_xy = pred_view[segment - 1]
            prev_quality_list, _ = self.get_quality(segment - 1, {}, video_xy, 0)
            avg_prev_quality = np.sum(prev_quality_list)
            oscillation_time = abs(avg_prev_quality - avg_quality)
        # 3. 计算空间平滑度
        oscillation_space = self.get_oscillation_space(avg_quality, quality_list_non)

        return avg_quality - LAMDA * (oscillation_time + oscillation_space)
