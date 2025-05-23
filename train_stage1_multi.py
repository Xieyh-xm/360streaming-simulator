import os
import time

import numpy as np
import torch
import random
from datetime import datetime
# from deep_rl.rl_env.rl_env import RLEnv
from deep_rl.rl_env.rl_env import RLEnv, STATE_DIMENSION, ACTION_DIMENSION
from deep_rl.ppo_multi import PPO
import math
import multiprocessing

net_trace = "./data_trace/network/generate"
state_dim = STATE_DIMENSION  # state space dimension
action_dim = ACTION_DIMENSION  # action space dimension

Alpha = 0.7

change_interval = 15  # 每隔15轮换一批验证集


def train():
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    # ============== Save Model ==============
    env_name = "lecture-stage-1"
    print("Training environment name : " + env_name)
    save_model_freq = 25  # save model frequency (in num timesteps)

    # =============== Hyper-parameter Setting ===============
    device = torch.device('cpu')
    # print("Device set to : ", device)

    K_epochs = 40  # update policy for K epochs in one PPO update
    # K_epochs = 60  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    # eps_clip = 0.1  # clip parameter for PPO
    gamma = 0.98  # discount factor

    # 起始300轮
    lr_actor = 0.00001  # learning rate for actor network
    lr_critic = 0.0005  # learning rate for critic network

    # 300轮后
    # lr_actor = 0.00003  # learning rate for actor network
    # lr_critic = 0.0001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    # net_trace = "./network/real_trace"

    # net_trace = "./data_trace/network/norway-test"
    # net_trace = "./network/generate"

    #  =============== Logging ===============
    log_dir = "deep_rl/ppo_logs"
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
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
    # ppo_agent.load("deep_rl/PPO_preTrained/stage_1_generate/PPO_stage_1_0_75.pth")

    # =============== 随机化trace ===============
    network_batch = 3
    # network_dict_size = 900  # 一阶段
    network_list = range(network_dict_size)

    video_batch = 2
    video_dict_size = 18
    video_list = range(video_dict_size)

    user_batch = 2
    user_dict_size = 48
    user_list = range(user_dict_size)

    ''' 第一阶段课程学习相关变量初始化 '''
    # beta = 0.3
    history_length = 2
    past_reward = None
    past_score = [[] for i in range(SUBTASK_NUM)]
    avg_increase = 0
    # ===========================================
    # training loop
    past_subtask_prob = [0 for i in range(SUBTASK_NUM)]
    past_subtask_prob[0] = 1
    while time_step <= max_training_timesteps:
        ticks = int(time.time())
        random.seed(ticks)
        ''' 1. 记录训练效果 '''
        train_net_id = None
        sum_train_net = 6  # 暂定5条trace
        if time_step % change_interval == 0:
            print(">>> 间隔15轮 --- 更新验证集 <<<")
            cur_reward, _ = test_in_validation(ppo_agent, time_step,
                                               seed=6)  # [subtask0,subtask1,...,subtask5]
            past_reward = cur_reward
            train_net_id, prev_sample_prob = organize_data(sum_train_net, past_subtask_prob)
            past_subtask_prob = prev_sample_prob
        else:
            cur_reward, cur_entropy = test_in_validation(ppo_agent, time_step,
                                                         seed=6)  # [subtask0,subtask1,...,subtask5]
            cur_delta_reward = np.sum(cur_reward - past_reward)

            score = calculate_score(avg_increase, cur_reward, past_reward, cur_entropy, past_subtask_prob)
            print("score = ", score)

            # avg_increase = cur_delta_reward
            avg_increase = (1 - Alpha) * avg_increase + Alpha * cur_delta_reward
            if abs(avg_increase - cur_delta_reward) > 2 * cur_delta_reward:
                avg_increase = cur_delta_reward

            print("cur_entropy = ", cur_entropy)
            past_reward = cur_reward  # save
            for task_id in range(SUBTASK_NUM):
                if len(past_score[task_id]) >= history_length:
                    past_score[task_id].pop(0)
                past_score[task_id].append(score[task_id])
            print("past_score = ", past_score)

            ''' 2. 组织训练数据 '''
            subtask_prob = generate_prob(past_score)

            print("subtask_prob = ", subtask_prob)
            train_net_id, prev_sample_prob = organize_data(sum_train_net, subtask_prob)
            past_subtask_prob = prev_sample_prob

        time_step += 1
        print(">>>>> train agent with sorted network")
        reward_q = multiprocessing.Queue()
        buffer_q = multiprocessing.Queue()
        train_jobs = []
        for net_id in train_net_id:
            ticks = int(time.time())
            random.seed(ticks)
            for video_id in random.sample(video_list, video_batch):
                for user_id in random.sample(user_list, user_batch):
                    p = multiprocessing.Process(target=sub_train,
                                                args=(ppo_agent, net_id, video_id, user_id, reward_q, buffer_q))
                    train_jobs.append(p)
                    p.start()

        for job in train_jobs:
            while job.is_alive():
                while not buffer_q.empty():
                    buffer = buffer_q.get()
                    ppo_agent.buffer.states += buffer[0]
                    ppo_agent.buffer.actions += buffer[1]
                    ppo_agent.buffer.logprobs += buffer[2]
                    ppo_agent.buffer.bw_masks += buffer[3]
                    ppo_agent.buffer.rewards += buffer[4]
                    ppo_agent.buffer.is_terminals += buffer[5]

        for p in train_jobs:
            p.join()
        sum_reward = 0

        for job in train_jobs:
            reward = reward_q.get()
            sum_reward += reward

        session_num = len(train_net_id) * video_batch * user_batch
        print_running_reward = sum_reward / session_num
        print("update network ...")
        ppo_agent.update()  # 更新模型
        print("\033[1;31m Episode : {} \t\t Timestep : {} \t\t Average Reward : {:.2f}\033[0m".format(i_episode,
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

    return


SUBTASK_NUM = 6
beta = 5


def calculate_score(avg_increase, cur_reward, past_reward, cur_entropy, past_data_prob):
    delta_reward = cur_reward - past_reward
    # 计算增益的贡献
    total_increase = np.sum(delta_reward)
    increase_reward = np.zeros(SUBTASK_NUM)
    for i in range(SUBTASK_NUM):
        increase_reward[i] = (total_increase - avg_increase) * past_data_prob[i]
    # 遗忘的任务需要多训练
    decrease_reward = np.abs(np.minimum(delta_reward, 0))
    # 动作不确定高的动作需要多训练
    score = increase_reward + decrease_reward + beta * cur_entropy
    score = np.clip(score, -300, 300, out=None)
    print("increase_reward = ", increase_reward)
    print("decrease_reward = ", decrease_reward)
    print("cur_entropy = ", cur_entropy)
    return score


def organize_data(sum_train_net, subtask_prob) -> list:
    ticks = int(time.time())
    random.seed(ticks)
    train_net_id = []
    # 按照概率抽样
    x = [i for i in range(SUBTASK_NUM)]
    prev_sample_prob = [0 for i in range(SUBTASK_NUM)]
    subtask_choice = np.random.choice(a=x, size=sum_train_net, replace=True, p=subtask_prob)
    per_num = network_dict_size // SUBTASK_NUM
    for i in range(len(subtask_choice)):
        task_id = subtask_choice[i]
        prev_sample_prob[task_id] += 1
        train_net_id += random.sample(range(task_id * per_num, (task_id + 1) * per_num), 1)

    prev_sample_prob = np.array(prev_sample_prob)
    prev_sample_prob = prev_sample_prob / np.sum(prev_sample_prob)
    # # 只取前k=3大的trace
    # subtask_prob = np.array(subtask_prob)
    # k = 3
    # selected_task = subtask_prob.argsort()[-k:]
    # selected_task = selected_task[::-1]  # 按照概率由小到大的方法保存数组序号
    # modified_prob = np.zeros(k)
    # for i in range(k):
    #     modified_prob[i] = subtask_prob[selected_task[i]]
    # modified_prob = modified_prob / np.sum(modified_prob)
    #
    # for i in range(len(selected_task)):
    #     task_id = selected_task[i]
    #     prob = modified_prob[i]
    #     per_num = network_dict_size // SUBTASK_NUM
    #     trace_num = min(math.ceil(sum_train_net * prob), sum_train_net - len(train_net_id))
    #     train_net_id += random.sample(range(task_id * per_num, (task_id + 1) * per_num), trace_num)

    print("train_net_id = ", train_net_id)
    return train_net_id, prev_sample_prob


# buffer [[],[],[]]
def sub_train(ppo_agent, net_id, video_id, user_id, reward_q, buffer_q):
    sub_env = RLEnv(net_trace)  # creat environment
    # 在 <特定网络><特定视频><特定用户> 播放一个视频
    state, bw_mask = sub_env.reset(net_id, video_id, user_id)
    sum_reward = 0
    action_cnt = 0
    done = False

    states = []
    actions = []
    logprobs = []
    bw_masks = []
    rewards = []
    is_terminals = []

    while not done:
        action_cnt += 1
        # 1. select action with policy
        action, _, action_probs = ppo_agent.select_action(state, bw_mask)  # 动作 & 动作的不确定度

        state_n = torch.zeros([state_dim])
        state_n[:] = state[0, :]
        bw_mask_n = torch.zeros([action_dim - 2])
        bw_mask_n[:] = torch.Tensor(bw_mask)

        states.append(state_n)
        actions.append(action)
        logprobs.append(action_probs)
        bw_masks.append(bw_mask_n)
        # 2. interacter with env
        state, bw_mask, reward, done = sub_env.step(action)
        # 3. saving reward and is_terminals
        rewards.append(reward)
        is_terminals.append(done)
        sum_reward += reward

    reward_q.put(sum_reward / action_cnt)
    buffer_q.put([states, actions, logprobs, bw_masks, rewards, is_terminals])


network_dict_size = 900


def test_in_validation(ppo_agent, time_step, seed=6):
    print(">>>>> test agent in validation")
    # 间隔15轮换一次验证集
    seed += time_step // change_interval
    random.seed(seed)
    # 分为6类，每类100种trace
    trace_in_subtask = network_dict_size // SUBTASK_NUM
    network_batch = 3
    net_list = random.sample(range(trace_in_subtask), network_batch)

    video_test_batch = 1
    video_dict_size = 18
    video_list = range(video_dict_size)

    user_test_batch = 3
    user_dict_size = 48
    user_list = range(user_dict_size)

    reward_list, entropy_list = [], []

    # 在每一个subtask中测试
    for subtask in range(SUBTASK_NUM):
        cur_net_list = [i + subtask * trace_in_subtask for i in net_list]
        multi_q = multiprocessing.Queue()
        test_jobs = []
        for net_id in cur_net_list:
            for video_id in random.sample(video_list, video_test_batch):
                for user_id in random.sample(user_list, user_test_batch):
                    p = multiprocessing.Process(target=test,
                                                args=(ppo_agent, net_id, video_id, user_id, multi_q))
                    test_jobs.append(p)
                    p.start()
                    # reward, entropy = test(ppo_agent, net_id, video_id, user_id)
                    # sum_reward += reward
                    # sum_entropy += entropy
        for p in test_jobs:
            p.join()

        sum_score, sum_entropy = 0, 0
        for job in test_jobs:
            avg_reward, avg_entropy = multi_q.get()
            sum_score += avg_reward
            sum_entropy += avg_entropy

        session_num = len(cur_net_list) * video_test_batch * user_test_batch
        reward_list.append(sum_score / session_num)
        entropy_list.append(sum_entropy / session_num)
    ppo_agent.buffer.clear()
    print("\033[1;36m avg score in validation = ", str(sum(reward_list) / len(reward_list)), "\033[0m")
    return np.array(reward_list), np.array(entropy_list)


def test(ppo_agent, net_id, video_id, user_id, multi_q):
    env = RLEnv(net_trace)  # creat environment
    state, bw_mask = env.reset(net_id, video_id, user_id)
    sum_entropy, sum_reward = 0, 0
    done = False
    action_cnt = 0
    while not done:
        action_cnt += 1
        action, action_entropy, _ = ppo_agent.select_action(state, bw_mask, argmax_flag=True)  # 动作 & 动作的不确定度
        state, bw_mask, reward, done = env.step(action)
        sum_reward += reward
        sum_entropy += action_entropy
    avg_reward = sum_reward / action_cnt
    avg_entropy = sum_entropy / action_cnt
    # return avg_reward, avg_entropy
    multi_q.put([avg_reward, avg_entropy])
    # env.close()


def generate_prob(past_score):
    ticks = int(time.time())
    random.seed(ticks)
    ret = []
    # 随机采样
    for task_id in range(len(past_score)):
        # score = random.choice(past_score[task_id])
        score = max(past_score[task_id])
        ret.append(score)
    # 归一化
    ret = np.array(ret)
    ret = softmax(ret)

    return ret


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


if __name__ == '__main__':
    train()
