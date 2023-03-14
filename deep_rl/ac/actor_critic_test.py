import math
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
from myPrint import print_debug


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        # todo: replace network
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # add
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            # nn.Softmax(dim=-1)
        )
        self.actor.to(device)

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # add
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.critic.to(device)

    def act(self, state, bw_mask):
        state_numpy = state.numpy()
        mask = np.zeros(self.action_dim)  # 初始化，都可以下，不下的掩蔽为-1e9

        '''=============== calculate mask ==============='''
        # 1. 基础层buffer要先于增强层buffer的下载
        bt_buffer_len = math.floor(state_numpy[0, 10])  # et的上限
        bit_level_et = 5  # et的码率等级
        mask[bt_buffer_len * bit_level_et:25] = 1
        # 2. ET层在prefetch阶段需要顺序加载
        # et_buffer_len = math.floor(state_numpy[0, 11])  # et层
        # mask[(et_buffer_len + 1) * bit_level_et:25] = 1
        # todo 3. 不对播放前来不及下载完成的segment进行ET层下载。
        ''' 测试掩蔽规则 '''
        for i in range(len(bw_mask)):
            mask[i] = max(bw_mask[i], mask[i])

        # 4. 不选中没有可更新数据的segment进行下载
        download_bit_bt = state_numpy[0, 12]  # bt待下载数据量
        if download_bit_bt == 0:
            mask[self.action_dim - 2] = 1
        else:
            mask[self.action_dim - 1] = 1  # todo:有可下载的bt，就不sleep
            # if bt_buffer_len < 15.0:  # bt 小于10，不sleep，可下载bt
            #     mask[self.action_dim - 1] = 1
            #     mask[self.action_dim - 2] = 0
            # else:  # bt 大于10，不下载bt，可sleep
            #     mask[self.action_dim - 2] = 1
            #     mask[self.action_dim - 1] = 0
            #
            ''' 测试掩蔽规则 '''
            # default 4.0
            if bt_buffer_len <= 4.0:  # 2. bt_buffer_len至少5个chunk
                mask[0:self.action_dim - 2] = 1
                mask[self.action_dim - 1] = 1

        download_bit_et = state_numpy[0, 13:18]  # et待下载数据量
        for segment_id in range(5):
            if download_bit_et[segment_id] == 0:
                mask[segment_id * bit_level_et:(segment_id + 1) * bit_level_et] = 1
        '''============================================='''
        print_debug("mask = ", mask)
        mask = torch.BoolTensor(mask).to(self.device)
        mask = mask.unsqueeze(0)
        # print(mask)
        ''' 动作掩蔽 '''
        # action_probs = F.softmax(self.actor(state) * mask, dim=-1)  # action masking
        # 将mask中为1的部分使用value替代（value通常是一个极大或极小值），0的部分保持原值
        action_probs = F.softmax(self.actor(state).masked_fill(mask, -1e9), dim=-1)
        dist = Categorical(action_probs)
        # action = dist.sample()
        action = action_probs.argmax()
        action_logprob = dist.log_prob(action)

        action_entropy = 0  # 计算动作熵
        for action in action_probs:
            prob = action.item()
            action_entropy -= prob * math.log2(prob)

        return action.detach(), action_logprob.detach(), action_entropy
