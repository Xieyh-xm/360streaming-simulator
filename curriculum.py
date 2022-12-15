import time
import random
from sabre360_with_qoe import Session
from utils import get_trace_file
import numpy as np
from myLog import myLog

# ============= Config Setting =============
default_config = {}
default_config['ewma_half_life'] = [4, 1]  # seconds
default_config['buffer_size'] = 5  # seconds
default_config['log_file'] = 'log/session.log'

network_batch = 5
network_dict_size = 600
network_list = range(network_dict_size)

video_batch = 4
video_dict_size = 18
video_list = range(video_dict_size)

user_batch = 4
user_dict_size = 48
user_list = range(user_dict_size)

mylog = myLog(path="log/lecture.log")
logger = mylog.get_log()

MAX_ENV = 30


class Curriculum:
    def __init__(self, net_trace, agent, env):
        self.net_trace = net_trace
        self.ppo_agent = agent
        self.env = env

        # todo:保存历史数据
        self.rule_qoe = np.zeros([network_dict_size, video_dict_size, user_dict_size], dtype=float)

        self.hard_network_id = []
        self.hard_video_id = []
        self.hard_pose_id = []
        self.cnt = 0

    def update_hard_env(self):
        logger.info("update_hard_env...")
        ''' 挑选困难环境 '''
        # 1. 随机sample一些trace测试
        chosen_network, chosen_video, chosen_user = self.random_choose_trace()
        cnt = 0
        for network_id in chosen_network:
            for video_id in chosen_video:
                for user_id in chosen_user:
                    cnt += 1
                    heuristic_qoe = self.test_rule_algro(network_id, video_id, user_id)
                    rl_qoe = self.test_rl_model(network_id, video_id, user_id)
                    gap = heuristic_qoe - rl_qoe  # 2. 计算gap
                    if gap > 5000.:  # 3. 加入hard list
                        # logger.info(
                        #     '<{}> hard env --> net : {}\t video : {}\t pose : {}\t gap = {:.3f}'.format(cnt, network_id,
                        #                                                                                 video_id,
                        #                                                                                 user_id, gap))
                        if len(self.hard_network_id) >= MAX_ENV:
                            self.pop_env()
                        self.hard_network_id.append(network_id)
                        self.hard_video_id.append(video_id)
                        self.hard_pose_id.append(user_id)
                    # elif gap > 0:
                    #     logger.info('<{}> rl-based model closed to heuristic rule =_='.format(cnt))
                    # else:
                    #     logger.info("<{}> rl-based model better than heuristic rule (*^▽^*)".format(cnt))
        if len(self.hard_network_id) == 0:
            return False
        return True

    def test_rl_model(self, network_id, video_id, user_id):
        ''' 测试rl model，返回平均 qoe '''
        config = default_config.copy()
        network_file, video_file, user_file = get_trace_file(self.net_trace, network_id, video_id, user_id)
        config['bandwidth_trace'] = network_file
        config['manifest'] = video_file
        config['pose_trace'] = user_file

        from deep_rl.solution_lecture import Melody
        config['abr'] = Melody

        session = Session(config)
        session.set_algro_agent(self.ppo_agent)  # 模型设置 使用当前的agent
        session.run()
        metrics = session.get_total_metrics()
        qoe = metrics[1]
        return qoe

    def test_rule_algro(self, network_id, video_id, user_id):
        ''' 测试启发式，返回平均 qoe '''
        if self.rule_qoe[network_id, video_id, user_id] != 0:
            return self.rule_qoe[network_id, video_id, user_id]

        config = default_config.copy()
        network_file, video_file, user_file = get_trace_file(self.net_trace, network_id, video_id, user_id)
        config['bandwidth_trace'] = network_file
        config['manifest'] = video_file
        config['pose_trace'] = user_file
        from abr.RAM360 import RAM360
        config['abr'] = RAM360
        # from abr.TTS import TTS
        # config['abr'] = TTS

        session = Session(config)
        session.run()
        metrics = session.get_total_metrics()

        qoe = metrics[1]
        self.rule_qoe[network_id, video_id, user_id] = qoe
        return qoe

    def random_choose_trace(self):
        logger.info("random_choose_trace...")
        ticks = int(time.time())
        random.seed(ticks)

        chosen_network = random.sample(network_list, network_batch)
        chosen_video = random.sample(video_list, video_batch)
        chosen_user = random.sample(user_list, user_batch)

        return chosen_network, chosen_video, chosen_user

    def curriculum_training(self):
        ''' 课程学习 '''
        self.ppo_agent.buffer.clear()
        sum_reward = 0.0
        cnt = 0
        for i in range(len(self.hard_network_id)):
            net_id = self.hard_network_id[i]
            video_id = self.hard_video_id[i]
            pose_id = self.hard_pose_id[i]
            state, bw_mask = self.env.reset(net_id, video_id, pose_id)
            done = False
            while not done:
                cnt += 1
                action = self.ppo_agent.select_action(state, bw_mask)
                state, bw_mask, reward, done = self.env.step(action)
                sum_reward += reward
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)
        self.ppo_agent.update()
        avg_reward = sum_reward / cnt
        return avg_reward

    def clear_hard_env(self):
        self.hard_network_id.clear()
        self.hard_video_id.clear()
        self.hard_pose_id.clear()

    def pop_env(self):
        self.hard_network_id.pop(0)
        self.hard_pose_id.pop(0)
        self.hard_video_id.pop(0)
