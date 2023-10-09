# 基于高斯分布生成trace

import random
import numpy as np
from network_process import NetworkTrace

DEFAULT_LATENCY = 20  # ms
TOTAL_TIME = 600  # s
PLAY_DURATION = 500  # ms


def generator(cur_scale, new_filename):
    new_trace = NetworkTrace()
    sample_num = int(TOTAL_TIME / (PLAY_DURATION / 1000.))
    avg_choose = [8000, 10000, 12000, 14000]
    avg = random.choice(avg_choose)
    for i in range(sample_num):
        delta_scale_choose = [-300, -150, 0, 150, 300]
        delta_scale = random.choice(delta_scale_choose)
        sample_bandwidth = np.random.normal(loc=avg, scale=cur_scale + delta_scale, size=None)
        new_trace.bandwidth.append(sample_bandwidth)
        new_trace.play_duration.append(PLAY_DURATION)
        new_trace.latency.append(DEFAULT_LATENCY)
    new_trace.plot_network(new_filename)
    new_trace.save_trace(PATH + new_filename)


SUBTASK_NUM = 6
TRACE_PER_SUBTASK = 150
# 5种标准差 -> [600,1200,1800,2400,3000]
PATH = "../data_trace/network/generate/"
if __name__ == '__main__':
    scale_list = [500, 1500, 2500, 3500, 4500, 5500]
    for i in range(SUBTASK_NUM):
        # 生成6种任务
        scale = scale_list[i]
        for j in range(TRACE_PER_SUBTASK):
            filename = "generate-{}.json".format(i * TRACE_PER_SUBTASK + j)
            generator(scale, filename)
