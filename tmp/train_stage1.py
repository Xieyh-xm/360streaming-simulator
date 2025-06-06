import os
import time

import numpy as np
import torch
import random
from datetime import datetime
# from deep_rl.rl_env.rl_env import RLEnv
from deep_rl.rl_env.rl_env import RLEnv
from deep_rl.ppo import PPO
from collections import deque

net_trace = "./data_trace/network/sorted_trace"
env = RLEnv(net_trace)  # creat environment

state_dim = env.get_state_dim()  # state space dimension
action_dim = env.get_action_dim()  # action space dimension


def train():
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    # ============== Save Model ==============
    env_name = "stage_1_generate"
    print("Training environment name : " + env_name)
    save_model_freq = 25  # save model frequency (in num timesteps)

    # =============== Hyper-parameter Setting ===============
    device = torch.device('cpu')
    print("Device set to : ", device)

    K_epochs = 80  # update policy for K epochs in one PPO update
    # K_epochs = 60  # update policy for K epochs in one PPO update

    # eps_clip = 0.2  # clip parameter for PPO
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.98  # discount factor

    # 起始300轮
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    # 300轮后
    # lr_actor = 0.00003  # learning rate for actor network
    # lr_critic = 0.0001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    # net_trace = "./network/real_trace"

    # net_trace = "./data_trace/network/norway-test"
    # net_trace = "./network/generate"

    #  =============== Logging ===============
    log_dir = "../deep_rl/ppo_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    # =============== Checkpointing ===============
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "deep_rl/PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    ''' ====================== Training Procedure ======================== '''
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("===========================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # todo: set up time_step
    time_step = 0
    i_episode = 0
    # ppo_agent.load(directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, time_step))

    # =============== 随机化trace ===============
    network_batch = 3
    # network_dict_size = 240  # generate 240
    # network_dict_size = 600  # real_trace & sorted_trace
    network_dict_size = 600  # 一阶段

    # network_dict_size = 290  # fcc 290
    # network_dict_size = 310  # norway 310
    network_list = range(network_dict_size)

    video_batch = 1
    video_dict_size = 18
    video_list = range(video_dict_size)

    user_batch = 1
    user_dict_size = 48
    user_list = range(user_dict_size)

    ''' 第一阶段课程学习相关变量初始化 '''
    beta = 0.3
    history_length = 5
    past_reward = None
    past_score = [[] for i in range(SUBTASK_NUM)]

    # ===========================================
    # training loop
    first_step = True
    while time_step <= max_training_timesteps:
        ticks = int(time.time())
        random.seed(ticks)
        ''' 1. 记录训练效果 '''
        train_net_id = []
        sum_train_net = 30  # 暂定30条trace
        if first_step:
            cur_reward, cur_entropy = test_in_validation(ppo_agent,
                                                         seed=6)  # [subtask0,subtask1,...,subtask5]
            past_reward = cur_reward
            first_step = False
            train_net_id = random.sample(network_list, sum_train_net)
        else:
            cur_reward, cur_entropy = test_in_validation(ppo_agent,
                                                         seed=6)  # [subtask0,subtask1,...,subtask5]
            delta_reward = cur_reward - past_reward
            score = abs(delta_reward) + beta * cur_entropy
            past_reward = cur_reward  # save
            for task_id in range(SUBTASK_NUM):
                if len(past_score[task_id]) > history_length:
                    past_score[task_id].pop(0)
                past_score[task_id].append(score[task_id])
            print("past_score = ", past_score)
            ''' 2. 组织训练数据 '''
            subtask_prob = generate_prob(past_score)
            print("subtask_prob = ", subtask_prob)
            for i in range(len(subtask_prob)):
                trace_num = int(sum_train_net * subtask_prob[i])
                train_net_id += random.sample(range(i * 100, (i + 1) * 100), trace_num)

        cur_ep_reward = 0
        time_step += 1
        session_num = 0  # 播放了几次
        print(">>>>> train agent with sorted network")
        for net_id in train_net_id:
            ticks = int(time.time())
            random.seed(ticks)
            for video_id in random.sample(video_list, video_batch):
                for user_id in random.sample(user_list, user_batch):
                    # 在 <特定网络><特定视频><特定用户> 播放一个视频
                    state, bw_mask = env.reset(net_id, video_id, user_id)
                    session_num += 1
                    action_cnt = 0
                    done = False

                    while not done:
                        action_cnt += 1
                        # 1. select action with policy
                        action, _ = ppo_agent.select_action(state, bw_mask)  # 动作 & 动作的不确定度
                        state, bw_mask, reward, done = env.step(action)
                        # 2. saving reward and is_terminals
                        ppo_agent.buffer.rewards.append(reward)
                        ppo_agent.buffer.is_terminals.append(done)
                        cur_ep_reward += reward / action_cnt

        # 在验证集上测试
        ppo_agent.update()  # 更新模型
        print_running_reward = cur_ep_reward / session_num

        avg_score = cur_ep_reward / session_num
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {:.2f}".format(i_episode,
                                                                                    time_step,
                                                                                    print_running_reward))
        log_f.write('{},{},{:.2f}\n'.format(i_episode, time_step, print_running_reward))
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, time_step)
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
        i_episode += 1
    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    return


SUBTASK_NUM = 6


def test_in_validation(ppo_agent, seed=6):
    print(">>>>> test agent in validation")
    random.seed(seed)
    # 分为6类，每类100种trace
    network_dict_size = 600
    trace_in_subtask = network_dict_size // SUBTASK_NUM
    network_batch = 3
    net_list = random.sample(range(trace_in_subtask), network_batch)

    video_test_batch = 1
    video_dict_size = 18
    video_list = range(video_dict_size)

    user_test_batch = 1
    user_dict_size = 48
    user_list = range(user_dict_size)

    reward_list, entropy_list = [], []

    # 在每一个subtask中测试
    for subtask in range(SUBTASK_NUM):
        sum_reward, sum_entropy = 0, 0
        cur_net_list = [i + subtask * trace_in_subtask for i in net_list]
        for net_id in cur_net_list:
            for video_id in random.sample(video_list, video_test_batch):
                for user_id in random.sample(user_list, user_test_batch):
                    reward, entropy = test(ppo_agent, net_id, video_id, user_id)
                    sum_reward += reward
                    sum_entropy += entropy
        session_num = len(cur_net_list) * len(video_list) * len(user_list)
        reward_list.append(sum_reward / session_num)
        entropy_list.append(sum_entropy / session_num)
    ppo_agent.buffer.clear()
    return np.array(reward_list), np.array(entropy_list)


def test(ppo_agent, net_id, video_id, user_id):
    state, bw_mask = env.reset(net_id, video_id, user_id)
    sum_entropy, sum_reward = 0, 0
    done = False
    action_cnt = 0
    while not done:
        action_cnt += 1
        action, action_entropy = ppo_agent.select_action(state, bw_mask, argmax_flag=True)  # 动作 & 动作的不确定度
        state, bw_mask, reward, done = env.step(action)
        sum_reward += reward
        sum_entropy += action_entropy
    avg_reward = sum_reward / action_cnt
    avg_entropy = sum_entropy / action_cnt
    return avg_reward, avg_entropy


def generate_prob(past_score):
    ticks = int(time.time())
    random.seed(ticks)
    ret = []
    sum_score = 0
    # 随机采样
    for task_id in range(len(past_score)):
        score = random.choice(past_score[task_id])
        ret.append(score)
        sum_score += score
    # 归一化
    for task_id in range(len(past_score)):
        ret[task_id] /= sum_score
    return ret


if __name__ == '__main__':
    train()
