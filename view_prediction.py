import numpy as np
import pandas as pd
from headset import headset
from sklearn.linear_model import Ridge, LinearRegression
from utils import Pose2VideoXY, get_tiles_in_viewport, calculate_viewing_proportion


class ViewportPrediction:
    def __init__(self):
        pass

    # returns list of (tile, weight) weighted between 0.0 and 1.0 (list does not necessarily include all tiles)
    def predict_tiles(self, segment_index):
        raise NotImplementedError


# todo: add a view prediction algorithm for testing environment
class TestPrediction(ViewportPrediction):
    def __init__(self, session_info, session_events, window_size=1000):
        self.session_info = session_info
        self.session_events = session_events
        self.manifest = session_info.get_manifest()

        self.cur_tiles = 0  # all tiles visible at current time
        self.cur_view = 0  # all tiles visible for some time since last segment transition

        self.cur_view_segment = 0  # we are playing inside self.cur_view_segment but not finished

        self.history_video_xy = []

        self.session_events.add_pose_handler(self.pose_event)
        self.session_events.add_play_handler(self.play_event)

    def pose_event(self, value1):
        (play_time, pose) = value1
        self.cur_tiles = headset.get_tiles(pose)
        self.cur_view |= self.cur_tiles
        # 保存头部运动轨迹
        (video_x, video_y) = Pose2VideoXY(pose)
        self.history_video_xy.append([play_time, video_x, video_y])  # play_time: ms

    def play_event(self, time):
        while self.session_info.get_buffer().get_played_segments() > self.cur_view_segment:
            self.session_info.get_log_file().log_str('view for segment=%d: %s' %
                                                     (self.cur_view_segment, headset.format_view(self.cur_view)))
            self.prev_view = self.cur_view
            self.cur_view = self.cur_tiles
            self.cur_view_segment += 1

    def build_model(self, pred_segment):
        if len(self.history_video_xy) == 0:
            return None, None
        # 1. 确定预测窗口&历史窗口大小
        history_pose = np.array(self.history_video_xy)
        latest_pose_time = history_pose[-1][0]  # ms
        pred_length = int(pred_segment * 1000 - latest_pose_time)  # 预测窗口 ms
        past_length = pred_length / 2  # 历史窗口
        # 2. 取出用于预测的数据并调整
        start_time = max(latest_pose_time - past_length, 0)  # 预测起点
        for idx in range(len(history_pose)):
            if start_time >= history_pose[idx][0]:
                break
        start_idx = idx  # 起始节点
        play_time = history_pose[start_idx:, 0]
        head_x, head_y = history_pose[start_idx:, 1], history_pose[start_idx:, 2]
        for i in range(1, len(head_x)):
            if head_x[i] - head_x[i - 1] >= 0.5:
                head_x[i] -= 1.0
            elif head_x[i] - head_x[i - 1] <= -0.5:
                head_x[i] += 1.0
        # 3. 预测
        play_time = np.array(play_time).reshape(len(play_time), 1)
        model_x = Ridge()
        # model_x = LinearRegression()
        model_x.fit(play_time, head_x)

        model_y = Ridge()
        # model_y = LinearRegression()
        model_y.fit(play_time, head_y)

        return model_x, model_y

    def build_model_to_get_acc(self, pred_length, pred_segment):
        if len(self.history_video_xy) == 0:
            return None, None
        # print("pred segment = {}".format(pred_segment))
        pred_length = int(pred_length * 1000.)
        past_length = pred_length / 2
        history_pose = np.array(self.history_video_xy)
        start_time = max(pred_segment * 1000 - pred_length - past_length, 0)
        end_time = max(pred_segment * 1000 - pred_length, 0)
        if end_time == 0:
            return None, None
        # 确定预测起点
        index = 0
        while history_pose[index][0] < start_time:
            index += 1
        start_index = index
        # 确定预测终点
        while index < len(history_pose) and history_pose[index][0] < end_time:
            index += 1
        end_index = index
        # 取出用于预测的数据并调整
        play_time = history_pose[start_index:end_index + 1, 0]
        head_x = history_pose[start_index:end_index + 1, 1]
        head_y = history_pose[start_index:end_index + 1, 2]
        for i in range(1, len(head_x)):
            if head_x[i] - head_x[i - 1] >= 0.5:
                head_x[i] -= 1.0
            elif head_x[i] - head_x[i - 1] <= -0.5:
                head_x[i] += 1.0
        # 预测
        play_time = np.array(play_time).reshape(len(play_time), 1)
        model_x = Ridge()
        # model_x = LinearRegression()
        model_x.fit(play_time, head_x)

        model_y = Ridge()
        # model_y = LinearRegression()
        model_y.fit(play_time, head_y)
        return model_x, model_y

    def predict_view(self, model_x, model_y, pred_segment):
        x_pred = model_x.predict([[pred_segment * 1000.]])
        if x_pred < 0:
            x_pred += 1
        elif x_pred >= 1:
            x_pred -= 1
        y_pred = model_y.predict([[pred_segment * 1000.]])

        # 和仿真环境保持一致，在y轴方向进行截断
        if y_pred > 1:
            y_pred = 1
        elif y_pred < 0:
            y_pred = 0
        return (float(x_pred), float(y_pred))

    def get_pred_acc(self, pred_pose, real_pose):
        ''' 计算视角预测的准确度 '''
        (x_pred, y_pred) = pred_pose
        (x_real, y_real) = real_pose
        pred_tile_list = get_tiles_in_viewport(x_pred, y_pred)
        real_tile_list = get_tiles_in_viewport(x_real, y_real)
        # 计算重合面积
        overlap_area = 0.0
        for tile_id in pred_tile_list:
            overlap_area += calculate_viewing_proportion(x_real, y_real, tile_id)

        # 计算总面积
        real_area = 0.0
        for tile_id in real_tile_list:
            real_area += calculate_viewing_proportion(x_real, y_real, tile_id)
        acc = overlap_area / real_area
        return acc
