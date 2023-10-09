import os
import time

import numpy as np
import torch
import random
from datetime import datetime
# from deep_rl.rl_env.rl_env import RLEnv
from deep_rl.rl_env.rl_env import RLEnv, STATE_DIMENSION, ACTION_DIMENSION
from deep_rl.ppo_multi import PPO
import multiprocessing

# net_trace = "./data_trace/network/sorted_trace"
net_trace = "./data_trace/network/generate"
state_dim = STATE_DIMENSION  # state space dimension
action_dim = ACTION_DIMENSION  # action space dimension

Alpha = 0.7


def train():
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    # ============== Save Model ==============
    env_name = "non-lecture"
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
    lr_actor = 0.0001  # learning rate for actor network
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
    # ppo_agent.load(directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, time_step))

    # =============== 随机化trace ===============
    network_batch = 3
    network_dict_size = 900  # 一阶段
    network_list = range(network_dict_size)

    video_batch = 3
    video_dict_size = 18
    video_list = range(video_dict_size)

    user_batch = 1
    user_dict_size = 48
    user_list = range(user_dict_size)

    while time_step <= max_training_timesteps:
        ticks = int(time.time())
        random.seed(ticks)
        ''' 1. 记录训练效果 '''
        time_step += 1
        reward_q = multiprocessing.Queue()
        buffer_q = multiprocessing.Queue()
        train_jobs = []
        for net_id in random.sample(network_list, network_batch):
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
                while False == buffer_q.empty():
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

        session_num = network_batch * video_batch * user_batch
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

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    return


SUBTASK_NUM = 5
beta = 2


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


if __name__ == '__main__':
    train()
    # past_score = [[120.16211369080486], [0.6151841050066044], [0.5701764034926123], [0.6088996317899383],
    #               [0.4576987014894281]]
    # ret = generate_prob(past_score)
    # print(ret)
