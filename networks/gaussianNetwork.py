#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GaussianActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim = None,
                 action_dim = None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()

        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(88, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(88, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(device)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
            dis_action = torch.tanh(action)
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': dis_action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}
    