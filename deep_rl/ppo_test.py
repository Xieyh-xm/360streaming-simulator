import torch
import torch.nn as nn
# from deep_rl.ac.actor_critic_et_update import ActorCritic
from deep_rl.ac.actor_critic_test import ActorCritic


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # =============== Hyper-parameter Setting ===============
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.buffer = RolloutBuffer()

        # =============== Actor-Critic Setting ===============
        self.policy = ActorCritic(self.state_dim, self.action_dim, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, bw_mask):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_probs, action_entropy = self.policy_old.act(state, bw_mask)

        state_n = torch.zeros([1, self.state_dim])
        state_n[0, :] = state[0, :]

        bw_mask_n = torch.zeros([1, self.action_dim - 2])
        bw_mask_n[0, :] = torch.Tensor(bw_mask)

        return action, action_entropy

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
