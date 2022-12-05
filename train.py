import os
import time
import torch
import random
from datetime import datetime
# from deep_rl.rl_env.rl_env import RLEnv
from deep_rl.rl_env.rl_env_bw_mask import RLEnv
from deep_rl.ppo import PPO
# from deep_rl.ppo_test import PPO


def train():
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    # ============== Save Model ==============
    env_name = "fcc"
    print("Training environment name : " + env_name)
    save_model_freq = 10  # save model frequency (in num timesteps)

    # =============== Hyper-parameter Setting ===============
    device = torch.device('cpu')
    print("Device set to : ", device)

    # K_epochs = 80  # update policy for K epochs in one PPO update
    K_epochs = 40  # update policy for K epochs in one PPO update

    # eps_clip = 0.2  # clip parameter for PPO
    eps_clip = 0.1  # clip parameter for PPO
    gamma = 0.95  # discount factor

    # 起始300轮
    # lr_actor = 0.0003  # learning rate for actor network
    # lr_critic = 0.001  # learning rate for critic network

    # 300轮后
    lr_actor = 0.00003  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    # net_trace = "./network/real_trace"
    net_trace = "./network/fcc-scaling"
    # net_trace = "./network/generate"
    env = RLEnv(net_trace)  # creat environment

    state_dim = env.get_state_dim()  # state space dimension
    action_dim = env.get_action_dim()  # action space dimension

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
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("===========================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # todo: set up time_step
    time_step = 740
    i_episode = 0
    ppo_agent.load(directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, time_step))

    # =============== 随机化trace ===============
    network_batch = 5
    # network_dict_size = 240  # generate 240
    # network_dict_size = 600   # real_trace
    network_dict_size = 290  # fcc 290
    # network_dict_size = 310  # norway 310
    network_list = range(network_dict_size)

    video_batch = 2
    video_dict_size = 18
    video_list = range(video_dict_size)

    user_batch = 2
    user_dict_size = 48
    user_list = range(user_dict_size)
    # ===========================================
    # train
    state, bw_mask = env.reset(1, 1, 1)
    done = False
    cnt = 0
    cur_ep_reward = 0
    while not done:
        cnt += 1
        # 1. select action with policy
        action = ppo_agent.select_action(state, bw_mask)
        state, bw_mask, reward, done = env.step(action)
        # 2. saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        cur_ep_reward += reward
    print_running_reward = cur_ep_reward / cnt
    avg_score = cur_ep_reward
    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {:.2f} \t Average Score: {:.2f}".format(i_episode,
                                                                                                         time_step,
                                                                                                         print_running_reward,
                                                                                                         avg_score))
    log_f.close()
    env.close()
    return


if __name__ == '__main__':
    train()
