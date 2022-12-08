import json
import os
import math
import numpy as np
import pandas as pd


def Pose2VideoXY(pose):
    '''
    将pose四元组转换为视频xy轴的坐标表示
    tiles_x: x轴上tile个数
    tiles_y: y轴上tile个数
    '''
    # 坐标转换
    (qx, qy, qz, qw) = pose
    x = 2 * qx * qz + 2 * qy * qw
    y = 2 * qy * qz - 2 * qx * qw
    z = 1 - 2 * qx * qx - 2 * qy * qy

    video_x = math.atan2(-z, x)
    if video_x < 0:
        video_x += 2 * math.pi
    video_x /= (2 * math.pi)  # 举例：video_x的范围就是0~1
    video_y = math.atan2(math.sqrt(x * x + z * z), y) / math.pi

    return (video_x, video_y)


class PoseTrace:
    def __init__(self, filename):
        self.filename = filename
        self.time_ms = []
        self.quaternion = []
        self.video_x = []
        self.video_y = []
        self.read_trace()

    def read_trace(self):
        with open(self.filename) as file:
            trace = json.load(file)
            for trace_info in trace:
                self.time_ms.append(trace_info['time_ms'])
                self.quaternion.append(trace_info['quaternion'])
        self.transform2xy()

    def transform2xy(self):
        for i in range(len(self.time_ms)):
            (x, y) = Pose2VideoXY(self.quaternion[i])
            self.video_x.append(x)
            self.video_y.append(y)

    def get_pose_std(self):
        ''' 5s的运动方差 '''
        video_x = np.array(self.video_x)
        video_y = np.array(self.video_y)

        start_time = self.time_ms[0]
        delta_time = 5000.
        index = 0
        start_idx = 0
        std_x, std_y = [], []
        while index < len(self.time_ms) - 1:
            while index < len(self.time_ms) - 1 and self.time_ms[index] - start_time < delta_time:
                index += 1
            end_time = self.time_ms[index]
            end_idx = index

            x = video_x[start_idx:end_idx]
            for i in range(1, x.shape[0]):
                if x[i] - x[i - 1] >= 0.5:
                    x[i] -= 1.0
                elif x[i] - x[i - 1] <= -0.5:
                    x[i] += 1.0
            y = video_y[start_idx:end_idx]

            std_x_per = np.std(x)
            std_x.append(std_x_per)
            std_y_per = np.std(y)
            std_y.append(std_y_per)
            # print(std_x, std_y)
            start_time = end_time
        std_x = np.array(std_x)
        std_y = np.array(std_y)
        return np.average(std_x), np.average(std_y)


def take_num(filename):
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ret = ""
    for ele in filename:
        if ele in num_list:
            ret += ele
    return int(ret)


ROOT_PATH = "../data_trace/video/pose_trace/"
if __name__ == '__main__':
    cnt = 0
    video_list = os.listdir(ROOT_PATH)
    video_list.sort(key=take_num)  # todo: 文件顺序
    video_csv, pose_csv = [], []
    std_x_csv, std_y_csv = [], []
    for video in video_list:
        pose_path = os.path.join(ROOT_PATH, video)
        pose_list = os.listdir(pose_path)
        pose_list.sort(key=take_num)
        for pose in pose_list:
            cnt += 1
            file_path = os.path.join(pose_path, pose)
            print(file_path)
            pose_trace = PoseTrace(file_path)
            avg_std_x, avg_std_y = pose_trace.get_pose_std()

            video_csv.append(video)
            pose_csv.append(pose)
            std_x_csv.append(avg_std_x)
            std_y_csv.append(avg_std_y)
            # print("{}\t{}\t{} \tstd_x = {:.3f} \tstd_y = {:.3f}".format(cnt, video, pose, avg_std_x, avg_std_y))
    data_frame = pd.DataFrame({'video': video_csv, 'pose': pose_csv, 'std_x': std_x_csv, 'std_y': std_y_csv})
    data_frame.to_csv("pose_info.csv", index=True, sep=',')
