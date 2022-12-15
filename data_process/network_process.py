import json
import os
import numpy as np

''' 用于放缩网络trace & 生成网络trace '''

DEFAULT_LATENCY = 20.


class NetworkTrace:
    ''' used for json file '''

    def __init__(self, filepath):
        self.file_path = filepath
        self.play_duration = []  # ms
        self.bandwidth = []  # kbps
        self.latency = []  # ms
        self.read_trace()

    def read_trace(self):
        with open(self.file_path) as file:
            trace = json.load(file)
            for trace_info in trace:
                self.play_duration.append(trace_info['duration_ms'])
                self.bandwidth.append(trace_info['bandwidth_kbps'])
                self.latency.append(trace_info['latency_ms'])

    def get_bw_avg(self):
        return sum(self.bandwidth) / len(self.bandwidth)  # kbps

    def get_bw_std(self):
        bw = np.array(self.bandwidth)
        return np.std(bw)

    def scale_bw_avg(self, target_avg):
        ''' 缩放trace '''
        cur_avg = self.get_bw_avg()
        ratio = target_avg / cur_avg
        for i in range(len(self.bandwidth)):
            self.bandwidth[i] *= ratio

    def save_trace(self, new_path):
        print("saving to {}".format(new_path))
        with open(new_path, 'w') as output_file:
            # todo:写入json文件
            data = []
            temp_dict = {}
            for i in range(len(self.play_duration)):
                temp_dict["duration_ms"] = self.play_duration[i]
                temp_dict["bandwidth_kbps"] = self.bandwidth[i]
                temp_dict["latency_ms"] = self.latency[i]
                data.append(temp_dict.copy())
            json.dump(data, output_file)


def MM_to_json(path):
    ''' MM Challenge trace to json file '''
    time_duration = []  # ms
    bitrate = []  # kbps
    latency = []  # ms
    last_playtime = -0.5
    # ========== 读取trace ==========
    with open(path, 'r') as trace_file:
        iter_f = iter(trace_file)
        for line in iter_f:
            idx = 0
            while line[idx] != ' ':
                idx += 1
            idx += 1
            # 1. record time duration
            playtime = float(line[:idx - 1]) * 1000.
            time_duration.append(playtime - last_playtime)
            last_playtime = playtime
            # 2. record bitrate
            if line[-1] == '\n':
                bitrate.append(float(line[idx:-1]) * 1024.)  # kbps
            else:
                bitrate.append(float(line[idx:]) * 1024.)
            # 3. record latency
            latency.append(DEFAULT_LATENCY)
    # ========== 输出至json文件 ==========
    data = []
    for i in range(len(time_duration)):
        tmp_dict = {}
        tmp_dict["duration_ms"] = time_duration[i]
        tmp_dict["bandwidth_kbps"] = bitrate[i]
        tmp_dict["latency_ms"] = latency[i]
        data.append(tmp_dict.copy())

    output_path = path + '.json'
    with open(output_path, 'w') as output_file:
        json.dump(data, output_file)


def Genet_to_json(path):
    ''' MM Challenge trace to json file '''
    time_duration = []  # ms
    bitrate = []  # kbps
    latency = []  # ms
    last_playtime = -0.5
    # ========== 读取trace ==========
    with open(path, 'r') as trace_file:
        iter_f = iter(trace_file)
        for line in iter_f:
            data = line.split()
            # 1. record time duration
            playtime = float(float(data[0]) * 1000.)
            time_duration.append(playtime - last_playtime)
            last_playtime = playtime
            # 2. record bitrate
            bitrate.append(float(data[1]) * 1024.)  # kbps
            # 3. record latency
            latency.append(DEFAULT_LATENCY)
    # ========== 输出至json文件 ==========
    data = []
    for i in range(len(time_duration)):
        tmp_dict = {}
        tmp_dict["duration_ms"] = time_duration[i]
        tmp_dict["bandwidth_kbps"] = bitrate[i]
        tmp_dict["latency_ms"] = latency[i]
        data.append(tmp_dict.copy())

    output_path = path + '.json'
    with open(output_path, 'w') as output_file:
        json.dump(data, output_file)


def trace2json(orgin):
    ''' 将已有trace转换为json格式 '''
    file_list = os.listdir(RAW_PATH)
    file_list.sort()
    for filename in file_list:
        if filename == ".DS_Store":
            continue
        file_path = RAW_PATH + filename
        if orgin == "MM":
            MM_to_json(file_path)
        elif orgin == "Genet":
            Genet_to_json(file_path)


def resize_json_trace(target_bw):
    ''' 按照目标平均码率缩放json trace '''
    file_list = os.listdir(RAW_PATH)
    file_list.sort()
    for filename in file_list:
        if filename == ".DS_Store":
            continue
        trace = NetworkTrace(filename)
        trace.read_trace()
        trace.scale_bw_avg(target_bw)
        trace.save_trace(NEW_PATH, new_info=str(target_bw) + "kbps")


def take_num(filename):
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ret = ""
    for ele in filename:
        if ele in num_list:
            ret += ele
    return int(ret)


def trace_classify():
    ''' 按照方差对trace进行分类 '''
    # # 1. 计算每个trace的方差
    # ROOT_PATH = "../data_trace/network/real_trace"
    # trace_name = os.listdir(ROOT_PATH)
    # trace_name.sort(key=take_num)
    # std_dict = {}  # 和trace_name一一对应
    # for filename in trace_name:
    #     path = os.path.join(ROOT_PATH, filename)
    #     trace = NetworkTrace(path)
    #     std_dict[filename] = trace.get_bw_std()
    #
    # # 2. 字典按值排序
    # trace_tuple = sorted(std_dict.items(), key=lambda x: x[1])
    # print(trace_tuple)
    #
    # # 3. 按顺序写入新的目录下
    # NEW_ROOT_PATH = "../data_trace/network/sorted_trace"
    # for i, ele in enumerate(trace_tuple):
    #     old_filename = ele[0]
    #     new_filename = "sorted_trace_{}.json".format(i)
    #     print(os.path.join(NEW_ROOT_PATH, new_filename))
    #     trace = NetworkTrace(os.path.join(ROOT_PATH, old_filename))
    #     trace.save_trace(os.path.join(NEW_ROOT_PATH, new_filename))

    # 4. 打印方差
    NEW_ROOT_PATH = "../data_trace/network/sorted_trace"
    sorted_trace = os.listdir(NEW_ROOT_PATH)
    sorted_trace.sort(key=take_num)
    for filename in sorted_trace:
        path = os.path.join(NEW_ROOT_PATH, filename)
        trace = NetworkTrace(path)
        print("filename : {}\t std = {}".format(filename, trace.get_bw_std()))


RAW_PATH = "../data_trace/network/norway-scaling/"
NEW_PATH = "../network/norway-9M/"
if __name__ == '__main__':
    # resize_json_trace(target_bw=9000)  # kbps
    # trace2json(orgin="Genet")
    trace_classify()
