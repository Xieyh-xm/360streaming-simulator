''' 用于测试的程序 '''
import random
from abr.myABR import TestAbr
from abr.RAM360 import RAM360
from deep_rl.solution import Melody
import numpy as np
from utils import get_trace_file
from sabre360_with_qoe import Session
from tqdm import tqdm

NETWORK_TRACE_NUM = 40
VIDEO_TRACE_NUM = 18
USER_TRACE_NUM = 48

# ============= Config Setting =============
default_config = {}
default_config['ewma_half_life'] = [4, 1]  # seconds
default_config['buffer_size'] = 5  # seconds
default_config['log_file'] = 'log/session.log'
# default_config['abr'] = TestAbr
# default_config['abr'] = Melody
default_config['abr'] = RAM360


def print_metrics(metrics):
    ''' ====== 打印各项指标 =================================
       [0]score [1]qoe [2]quality [3]stall_time [4]var_space
       [5]var_time [6]bandwidth_usage [7]bandwidth_wastage'''
    print("------------------------------------------------------------------")
    print('Score: {:.2f}'.format(metrics[0]))
    print('QoE: {:.2f}\t\tbandwidth_usage: {:.2f}'.format(metrics[1], metrics[6]))
    print('Quality: {:.2f}\tStall time: {:.2f}\t'.format(metrics[2], metrics[3]))
    print('Oscillation in space: {:.2f}\tOscillation in time: {:.2f}'.format(metrics[4], metrics[5]))
    wastage_ratio = metrics[7] / metrics[6]
    print('Wastage ratio:{:.2f}'.format(wastage_ratio))


def test(net_id, video_id, user_id):
    # set config
    config = default_config.copy()
    network_file, video_file, user_file = get_trace_file(net_id, video_id, user_id)
    config['bandwidth_trace'] = network_file
    config['manifest'] = video_file
    config['pose_trace'] = user_file
    # print("------------------------------------------------------------------")
    # print("network trace: {}\nvideo trace: {}\nuser trace: {}".format(network_file, video_file, user_file))
    session = Session(config)
    session.run()
    avgs = session.get_total_metrics()
    print_metrics(avgs)
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
    for net_id in tqdm(random.sample(network_list, network_batch)):
        avgs += test_video_samples(net_id, video_batch, user_batch)
    avgs /= network_batch
    print_metrics(avgs)


if __name__ == '__main__':
    # test_network_samples(network_batch=10, video_batch=5, user_batch=5)
    test(0, 0, 0)
