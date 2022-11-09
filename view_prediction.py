import numpy as np
import math
from collections import deque
from headset import headset
import navigation.navigation_graph as ng
from sklearn.linear_model import Ridge
from utils import Pose2VideoXY


class ViewportPrediction:
    def __init__(self):
        pass

    # returns list of (tile, weight) weighted between 0.0 and 1.0 (list does not necessarily include all tiles)
    def predict_tiles(self, segment_index):
        raise NotImplementedError


class NavigationGraphPrediction(ViewportPrediction):
    def __init__(self, session_info, session_events, navigation_graph_path):
        self.session_info = session_info
        self.session_events = session_events
        self.manifest = session_info.get_manifest()

        self.single_graph = ng.SUNavigationGraph()
        self.cross_graph = ng.CUNavigationGraph(navigation_graph_path)

        self.cur_tiles = 0  # all tiles visible at current time
        self.cur_view = 0  # all tiles visible for some time since last segment transition

        self.prev_view = None  # when we finished playing last segment, it had self.prev_view
        self.cur_view_segment = 0  # we are playing inside self.cur_view_segment but not finished
        self.single_memo_prediction = []
        self.cross_memo_prediction = []

        # self.choosing_cross_user = True
        self.choosing_cross_user = False
        self.next_prediction_cross = None
        self.next_prediction_single = None

        self.session_events.add_pose_handler(self.pose_event)
        self.session_events.add_play_handler(self.play_event)

    def pose_event(self, pose):
        self.cur_tiles = headset.get_tiles(pose)
        self.cur_view |= self.cur_tiles

    def play_event(self, time):
        # It is important to use get_played_segment() here and not get_play_head()
        # because otherwise rounding errors might cause some issues elsewhere.
        while self.session_info.get_buffer().get_played_segments() > self.cur_view_segment:
            self.session_info.get_log_file().log_str('view for segment=%d: %s' %
                                                     (self.cur_view_segment, headset.format_view(self.cur_view)))

            # check prediction to compare precision for single-user and cross-user
            if self.cur_view_segment > 1:
                # we can only use single-user prediction after first segment
                single_prediction = self.predict_single(self.cur_view_segment)
                assert (single_prediction is not None)
                single_precision = ng.check_precision(self.cur_view, single_prediction)

                cross_prediction = self.predict_cross(self.cur_view_segment)
                assert (cross_prediction is not None)
                cross_precision = ng.check_precision(self.cur_view, cross_prediction)

                self.choosing_cross_user = cross_precision >= single_precision

                self.session_info.get_log_file().log_str(
                    'navigation graph precision: segment=%d, single precision = %.3f, cross precision = %.3f, choosing %s' %
                    (self.cur_view_segment, single_precision, cross_precision,
                     'cross' if self.choosing_cross_user else 'single'))
            else:
                cross_prediction = self.predict_cross(self.cur_view_segment)
                assert (cross_prediction is not None)
                cross_precision = ng.check_precision(self.cur_view, cross_prediction)
                self.session_info.get_log_file().log_str(
                    'navigation graph precision: segment=%d, single precision = -, cross precision = %.3f, choosing %s' %
                    (self.cur_view_segment, cross_precision, 'cross'))

            # we get a new view, so we will need to recalculate all predictions
            self.single_memo_prediction = []
            self.cross_memo_prediction = []

            # update single-user graph
            if self.prev_view is not None:  # cannot insert startup entry
                self.single_graph.update(self.cur_view, self.prev_view)

            self.prev_view = self.cur_view
            self.cur_view = self.cur_tiles
            self.cur_view_segment += 1

    def predict_single(self, segment):
        memo_index = segment - self.cur_view_segment
        if memo_index < 0:
            # don't predict the past
            return None

        if memo_index < len(self.single_memo_prediction):
            return self.single_memo_prediction[memo_index][1]

        if self.cur_view_segment < 2:
            # we have the first graph after having two segment transitions done
            return None

        if len(self.single_memo_prediction) == 0:
            assert (self.prev_view is not None)
            view_vector = self.single_graph.view_to_view_vector(self.prev_view)
            assert (view_vector is not None)
        else:
            view_vector = self.single_memo_prediction[-1][0]

        while len(self.single_memo_prediction) <= memo_index:
            view_vector = self.single_graph.predict(view_vector)
            tile_vector = self.single_graph.view_vector_to_tile_vector(view_vector)
            self.single_memo_prediction += [(view_vector, tile_vector)]

        return self.single_memo_prediction[memo_index][1]

    def predict_cross(self, segment):
        memo_index = segment - self.cur_view_segment
        if memo_index < 0:
            # don't predict the past
            return None

        if memo_index < len(self.cross_memo_prediction):
            return self.cross_memo_prediction[memo_index][1]

        if len(self.cross_memo_prediction) == 0:
            if self.cur_view_segment == 0:
                # we have not seen any view information yet
                assert (self.prev_view is None)
                view_vector = np.ones(1)
            else:
                assert (self.prev_view is not None)
                view_vector = self.cross_graph.view_to_view_vector(self.cur_view_segment - 1, self.prev_view)
                assert (view_vector is not None)
        else:
            view_vector = self.cross_memo_prediction[-1][0]

        while len(self.cross_memo_prediction) <= memo_index:
            segment = self.cur_view_segment + len(self.cross_memo_prediction)
            view_vector = self.cross_graph.predict(segment, view_vector)
            tile_vector = self.cross_graph.view_vector_to_tile_vector(segment, view_vector)
            self.cross_memo_prediction += [(view_vector, tile_vector)]

        return self.cross_memo_prediction[memo_index][1]

    def predict_tiles(self, segment):
        if self.choosing_cross_user:
            return self.predict_cross(segment)
        else:
            return self.predict_single(segment)


# todo: add a view prediction algorithm for testing environment
class TestPrediction(ViewportPrediction):
    def __init__(self, session_info, session_events, window_size=1000):
        self.session_info = session_info
        self.session_events = session_events
        self.manifest = session_info.get_manifest()

        self.cur_tiles = 0  # all tiles visible at current time
        self.cur_view = 0  # all tiles visible for some time since last segment transition

        self.cur_view_segment = 0  # we are playing inside self.cur_view_segment but not finished

        self.history_video_xy = deque(maxlen=window_size)

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
        model_x.fit(play_time, head_x)

        model_y = Ridge()
        model_y.fit(play_time, head_y)

        return model_x, model_y

    def predict_view(self, model_x, model_y, pred_segment):
        x_pred = model_x.predict([[pred_segment * 1000]])
        if x_pred < 0:
            x_pred += 1
        elif x_pred >= 1:
            x_pred -= 1
        y_pred = model_y.predict([[pred_segment * 1000]])

        # 和仿真环境保持一致，在y轴方向进行截断
        if y_pred > 1:
            y_pred = 1
        elif y_pred < 0:
            y_pred = 0
        return (float(x_pred), float(y_pred))
