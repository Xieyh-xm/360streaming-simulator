''' 用于测试的程序 '''
import random

import numpy as np
from utils import get_trace_file

NETWORK_TRACE_NUM = 40
VIDEO_TRACE_NUM = 18
USER_TRACE_NUM = 48


def test(net_id, video_id, user_id):
    avgs = np.zeros(8)
    return avgs


def test_user_samples(net_id, video_id, user_batch=USER_TRACE_NUM):
    avgs = np.zeros(8)
    user_list = range(USER_TRACE_NUM)
    for user_id in random.sample(user_list, user_batch):
        avgs += test(net_id, video_id, user_id)
    avgs /= user_batch
    return avgs


def test_video_samples(net_id, video_batch=VIDEO_TRACE_NUM, user_batch=USER_TRACE_NUM):
    avgs = np.zeros(8)
    video_list = range(VIDEO_TRACE_NUM)
    for video_id in random.sample(video_list, video_batch):
        avgs += test_user_samples(net_id, video_id, user_batch)
    avgs /= video_batch
    return avgs


def test_network_samples(network_batch=NETWORK_TRACE_NUM, video_batch=VIDEO_TRACE_NUM, user_batch=USER_TRACE_NUM):
    avgs = np.zeros(8)  # [0]score [1]qoe [2]quality [3]stall_time [4]var_space [5]var_time [6]bandwidth_usage
    network_list = range(NETWORK_TRACE_NUM)
    for net_id in random.sample(network_list, network_batch):
        avgs += test_video_samples(net_id, video_batch, user_batch)
    avgs /= network_batch

    print("Score: {}".format(avgs[0]))
    print("QoE: {}".format(avgs[1]))
    print("Quality: {}".format(avgs[2]))
    print("Stall time: {}".format(avgs[3]))
    print("Oscillation in time: {}".format(avgs[4]))
    print("Oscillation in space: {}".format(avgs[5]))
    print("Bandwidth usage: {}".format(avgs[6]))
    print("Bandwidth wastage: {}".format(avgs[7]))


if __name__ == '__main__':
    test_network_samples()
