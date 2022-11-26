import math
from collections import namedtuple
import numpy as np
from utils import get_trace_file, get_tiles_in_viewport, calculate_viewing_proportion, Pose2VideoXY
from abr.TiledAbr import TiledAbr
from sabre360_with_qoe import SessionInfo, SessionEvents

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


class RAM360(TiledAbr):
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
        self.qoe_estimator = QoeEstimator(TILES_X, TILES_Y, self.session_info.get_buffer(), self.bitrate)

        self.last_action = None

    def get_action(self):
        # 变量更新
        first_segment = self.buffer.get_played_segments()
        first_segment_offset = self.buffer.get_played_segment_partial()

        first_unplayed_segment = first_segment  # l_{k}
        unplayed_segment_num = self.buffer.get_buffer_depth()  # L_{k}
        if first_segment_offset != 0:
            first_unplayed_segment = min(first_unplayed_segment + 1, self.max_segment_id)
            unplayed_segment_num -= 1
        max_download_segment_id = min(first_unplayed_segment + unplayed_segment_num, self.max_segment_id)

        acc = self.calculate_pred_acc()  # 更新视角预测结果
        tput = self.calculate_throughput(self.last_action)

        max_utility = float("-inf")
        optimal_action = None
        model_x, model_y = self.session_info.get_viewport_predictor().build_model(max_download_segment_id)
        for segment_id in range(first_unplayed_segment, max_download_segment_id + 1):  # 遍历可下载的每一个segment
            '''============= 视点预测 ============='''
            if model_x is None:
                self.pred_view[segment_id] = (0.5, 0.5)
            else:
                self.pred_view[segment_id] = self.session_info.get_viewport_predictor().predict_view(model_x, model_y,
                                                                                                     segment_id)
            if self.pred_view[segment_id] is not None:
                (x_pred, y_pred) = self.pred_view[segment_id]
            else:
                x_pred, y_pred = 0.5, 0.5
            tiles_in_viewport = get_tiles_in_viewport(x_pred, y_pred)  # 视窗内的tile

            cur_action = None
            utility = float("-inf")
            if segment_id < max_download_segment_id:
                '''============= ET:update ============='''
                download_list = {}
                download_time = 0.0
                download_bit = 0.0
                stall_download_time = self.segment_duration * (unplayed_segment_num - STALL_TH) - first_segment_offset
                valid_download_time = self.segment_duration * (segment_id - first_segment + 1 - BANDWIDTH_TH) \
                                      - first_segment_offset
                for tile_id in tiles_in_viewport:  # 遍历视窗内的tile
                    max_tile_utility = float("-inf")
                    best_level = None
                    for bit_level in range(1, len(self.bitrate)):  # 遍历每个码率等级
                        ''' 限制条件 '''
                        # 1. 避免播放卡顿
                        tile_size = self.video_size[segment_id][tile_id][bit_level]  # bits
                        tmp_download_time = download_time + tile_size / tput  # ms
                        if tmp_download_time > stall_download_time:
                            break
                        # 2. 避免无效更新
                        if tmp_download_time > valid_download_time:
                            break
                        # 3. 避免重复下载同一码率的tile
                        cur_level = self.buffer.get_buffer_element(segment_id, tile_id)
                        if cur_level is not None and bit_level <= cur_level:
                            continue
                        ''' 计算效用函数 '''
                        tmp_download_list = download_list.copy()
                        tmp_download_list[tile_id] = bit_level

                        pred_accuracy = acc[min(segment_id - first_segment, len(acc) - 1)]
                        old_quality = self.qoe_estimator.get_total_quality(segment_id, {}, self.pred_view,
                                                                           pred_accuracy)
                        new_quality = self.qoe_estimator.get_total_quality(segment_id, tmp_download_list,
                                                                           self.pred_view,
                                                                           pred_accuracy)
                        quality_improvement = new_quality - old_quality
                        tmp_download_bit = download_bit + tile_size

                        assert tmp_download_bit > 0
                        tile_utility = CONTROL_PARM_V * quality_improvement / (tmp_download_bit / 1024.)
                        if tile_utility > max_tile_utility:
                            max_tile_utility = tile_utility
                            best_level = bit_level
                    if best_level is not None:
                        download_list[tile_id] = best_level
                        download_bit += self.video_size[segment_id][tile_id][best_level]
                        download_time += self.video_size[segment_id][tile_id][best_level] / tput  # ms

                cur_action = AvailableAction(segment_id, download_list)
                if len(download_list) != 0:
                    pred_accuracy = acc[min(segment_id - first_segment, len(acc) - 1)]
                    old_quality = self.qoe_estimator.get_total_quality(segment_id, {}, self.pred_view,
                                                                       pred_accuracy)
                    new_quality = self.qoe_estimator.get_total_quality(segment_id, download_list, self.pred_view,
                                                                       pred_accuracy)
                    quality_improvement = new_quality - old_quality
                    assert download_bit > 0
                    utility = CONTROL_PARM_V + quality_improvement / (download_bit / 1024.)
                else:  # 该segment没有tile可以下载
                    utility = float("-inf")
            elif segment_id == max_download_segment_id:
                '''============= BT:prefetch ============='''
                # 不重复下载
                if self.buffer.get_buffer_contents(segment_id) is not None:
                    continue
                download_bit = 0.0  # b_{k}
                download_tile = {}
                for i in range(TILES_X * TILES_Y):  # 全画幅下载
                    download_bit += self.video_size[segment_id][i][0]
                    download_tile[i] = 0
                buffer_length = self.buffer.get_buffer_depth() - first_segment_offset / 1000.  # Q_{k} ms

                cur_action = AvailableAction(segment=segment_id, download_tile_dict=download_tile)
                # I_{k}的计算
                old_quality = self.qoe_estimator.get_total_quality(segment_id, {}, self.pred_view, 1)
                new_quality = self.qoe_estimator.get_total_quality(segment_id, download_tile, self.pred_view, 1)
                quality_improvement = new_quality - old_quality
                assert download_bit > 0
                utility = (CONTROL_PARM_V * quality_improvement - buffer_length * self.segment_duration) / (
                        download_bit / 1024.)

            # print(utility)
            if utility > max_utility:
                max_utility = utility
                optimal_action = cur_action

        # 将 AvailableAction 转换为 TiledAction
        if optimal_action is None:
            return [TiledAction(0, 0, 0, 150)]
        optimal_segment = optimal_action.segment
        optimal_download_tile = optimal_action.download_tile_dict
        action = []
        for tile_id in optimal_download_tile:
            action.append(TiledAction(optimal_segment, tile_id, optimal_download_tile[tile_id], 0))
        self.last_action = action
        # print(action)
        return action

    def calculate_pred_acc(self):
        ''' 返回最新的视角预测精度 '''
        acc = DEFAULT_PRED_ACC.copy()
        pose_trace = self.session_info.get_prev_pose_trace()
        if len(pose_trace) == 0:
            return acc
        latest_segment = list(pose_trace.keys())[-1]
        latest_pose = pose_trace.get(list(pose_trace.keys())[-1])
        real_pose = Pose2VideoXY(latest_pose)
        for i in range(len(DEFAULT_PRED_ACC)):
            pred_len = i + 1
            model_x, model_y = self.view_predictor.build_model_to_get_acc(pred_len, latest_segment)
            if model_x is None:
                continue
            pred_pose = self.view_predictor.predict_view(model_x, model_y, latest_segment)
            pred_acc = self.view_predictor.get_pred_acc(pred_pose, real_pose)
            acc[i] = pred_acc
        return np.array(acc)

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


class QoeEstimator:
    def __init__(self, tiles_x, tiles_y, buffer, bitrate):
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.tiles_num = tiles_x * tiles_y
        self.buffer = buffer
        self.bitrate = bitrate

    def get_oscillation_space(self, avg_quality, quality_list_non, video_xy):
        ''' 计算segment空间平滑度(非差值 & 面积平均) '''
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
                download_quality = self.bitrate[download_tiles[tile_id]]
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
        return quality_list, quality_non_list

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
        oscillation_space = self.get_oscillation_space(avg_quality, quality_list_non, video_xy)

        # print("avg_quality = {:.2f}\toscillation_time = {:.2f}\toscillation_space = {:.2f})".format(avg_quality,
        #                                                                                             oscillation_time,
        #                                                                                             oscillation_space))
        # print("quality = {:.2f}\n".format(avg_quality - LAMDA * (oscillation_time + oscillation_space)))
        return avg_quality - LAMDA * (oscillation_time + oscillation_space)
