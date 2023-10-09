import os
import time
import torch
import random
from datetime import datetime
from deep_rl.rl_env.rl_env import RLEnv
from deep_rl.ppo import PPO
from curriculum import Curriculum
from myLog import myLog


def train():
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    # ============== Save Model ==============
    env_name = "genet"
    print("Training environment name : " + env_name)
    save_model_freq = 10  # save model frequency (in num timesteps)

    # =============== Hyper-parameter Setting ===============
    device = torch.device('cpu')
    print("Device set to : ", device)

    # K_epochs = 80  # update policy for K epochs in one PPO update
    K_epochs = 80  # update policy for K epochs in one PPO update

    # eps_clip = 0.2  # clip parameter for PPO
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.98  # discount factor

    # 300轮后
    lr_actor = 0.00001  # learning rate for actor network
    lr_critic = 0.00003  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    net_trace = "./data_trace/network/generate"
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
    # log_f = open(log_f_name, "w+")
    # log_f.write('episode,timestep,reward\n')

    time_step = 0
    i_episode = 0
    ppo_agent.load("deep_rl/PPO_preTrained/non-lecture/PPO_init-generate_0_225.pth")

    # ===========================================
    # mylog = myLog(path="20221206-lecture.log")
    lecture = Curriculum(net_trace, ppo_agent, env)
    REPEAT_NUM = 1
    # training loop
    while time_step <= max_training_timesteps:
        hard_flag = lecture.update_hard_env()
        if not hard_flag:  # 不存在困难环境
            print("Does not exist hard environment.")
            continue
        avg_reward = 0
        print("Start training ...")
        for i in range(REPEAT_NUM):
            avg_reward = lecture.curriculum_training()
        time_step += 1
        print_running_reward = avg_reward
        print("\033[1;31m Episode : {} \t\t Timestep : {} \t\t Average Reward : {:.2f}\033[0m".format(i_episode,
                                                                                                      time_step,
                                                                                                      print_running_reward))
        # log_f.write('{},{},{:.2f}\n'.format(i_episode, time_step, print_running_reward))
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
    # log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    return


if __name__ == '__main__':
    train()
