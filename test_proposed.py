''' 用于测试的程序 '''
import random
import numpy as np
from utils import get_trace_file, print_metrics, print_to_csv
from sabre360_with_qoe import Session
from tqdm import tqdm

# net = "norway-9M"
# net = "fcc-9M"
# net = "4g-logs"
# net = "4g-scaling"
# net = "fcc-scaling"
# net = "norway-scaling"
# net = "fcc-test"
net = "norway-test"
if net == "fcc-scaling":
    net_trace = "./data_trace/network/fcc-scaling"
    NETWORK_TRACE_NUM = 290
elif net == "norway-scaling":
    net_trace = "./data_trace/network/norway-scaling"
    NETWORK_TRACE_NUM = 310
elif net == "4g-scaling":
    net_trace = "./data_trace/network/4g-scaling"
    NETWORK_TRACE_NUM = 40
elif net == "4g-logs":
    net_trace = "./data_trace/network/raw_trace/4Glogs"
    NETWORK_TRACE_NUM = 40
elif net == "fcc-9M":
    net_trace = "./data_trace/network/fcc-9M"
    NETWORK_TRACE_NUM = 290
elif net == "norway-9M":
    net_trace = "./data_trace/network/norway-9M"
    NETWORK_TRACE_NUM = 310
elif net == "norway-test":
    net_trace = "./data_trace/network/norway-test"
    NETWORK_TRACE_NUM = 20
elif net == "fcc-test":
    net_trace = "./data_trace/network/fcc-test"
    NETWORK_TRACE_NUM = 20

VIDEO_TRACE_NUM = 18
USER_TRACE_NUM = 48

# ============= Config Setting =============
default_config = {}
default_config['ewma_half_life'] = [4, 1]  # seconds
default_config['buffer_size'] = 5  # seconds
default_config['log_file'] = 'log/session.log'

# ================= 测试算法 =================
# TestABR = "RAM360"
# TestABR = "TTS"
TestABR = "Melody"
if TestABR == "RAM360":
    from abr.RAM360 import RAM360

    default_config['abr'] = RAM360
elif TestABR == "TTS":
    from abr.TTS import TTS

    default_config['abr'] = TTS
elif TestABR == "Melody":
    from deep_rl.solution import Melody

    default_config['abr'] = Melody


def test(net_id, video_id, user_id):
    # set config
    config = default_config.copy()
    network_file, video_file, user_file = get_trace_file(net_trace, net_id, video_id, user_id)
    # print("net id  = {}\t video id = {}\t user id = {}".format(net_id, video_id, user_id))
    config['bandwidth_trace'] = network_file
    config['manifest'] = video_file
    config['pose_trace'] = user_file
    # print("------------------------------------------------------------------")
    # print("network trace: {}\nvideo trace: {}\nuser trace: {}".format(network_file, video_file, user_file))
    session = Session(config)
    session.run()
    avgs = session.get_total_metrics()
    # print_metrics(avgs)
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
    print_to_csv(avgs, TestABR, net)


random.seed(10)
if __name__ == '__main__':
    test_network_samples(network_batch=20, video_batch=4, user_batch=5)
    # test(4, 2, 2)
