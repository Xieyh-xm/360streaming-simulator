import os
import pandas as pd
import matplotlib.pyplot as plt

from network_process import NetworkTrace

NETWORK_PATH = "../data_trace/network/raw_trace/fcc"


# NETWORK_PATH = "../data_trace/network/norway-scaling"


# find ./ -name ".DS_Store" -depth -exec rm {} \;

def take_num(filename):
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ret = ""
    for ele in filename:
        if ele in num_list:
            ret += ele
    return int(ret)


def show_bw_info():
    file_list = os.listdir(NETWORK_PATH)
    file_list.sort(key=take_num)
    sum_std = 0
    avg, std = [], []
    for file in file_list:
        file_path = os.path.join(NETWORK_PATH, file)
        print(file_path)
        trace = NetworkTrace(file_path)
        trace.read_trace()
        avg.append(trace.get_bw_avg())
        std.append(trace.get_bw_std())

        sum_std += trace.get_bw_std()
    avg_std = sum_std / len(file_list)
    print("avg_std = {}".format(avg_std))
    data_frame = pd.DataFrame({'bw_avg': avg, 'bw_std': std})
    data_frame.to_csv("fcc.csv", index=True, sep=',')


def show_trace():
    file_path = "../data_trace/network/norway-scaling/norway_scaling-19.json"
    trace = NetworkTrace(file_path)
    playtime = []
    bandwidth = []
    cur_time = 0
    for i in range(len(trace.play_duration)):
        cur_time += trace.play_duration[i]
        playtime.append(cur_time)
        bandwidth.append(trace.bandwidth[i])
    plt.plot(playtime, bandwidth)
    plt.title('norway_scaling-19')
    plt.xlabel('time (ms)')
    plt.ylabel('bw (kbps)')
    plt.show()


if __name__ == '__main__':
    show_bw_info()
    # show_trace()
