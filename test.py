''' 用于测试的程序 '''
import numpy as np


def test_user_samples(sample_cnt):
    avgs = 0
    return avgs


def test_video_samples(sample_cnt):
    avgs = 0
    return avgs


def test_network_samples(sample_cnt):
    avgs = np.zeros(7)  # [0]score [1]qoe [2]quality [3]stall_time [4]var_space [5]var_time [6]bandwidth_usage
    for i in range(sample_cnt):
        pass

    print("Score: {}".format(avgs[0]))
    print("QoE: {}".format(avgs[1]))
    print("Quality: {}".format(avgs[2]))
    print("Stall time: {}".format(avgs[3]))
    print("Oscillation in time: {}".format(avgs[4]))
    print("Oscillation in space: {}".format(avgs[5]))
    print("Bandwidth usage:{}".format(avgs[6]))


if __name__ == '__main__':
    test_network_samples()
