import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.optim import Adam

import numpy as np

class Actor(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Actor, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> dim_actions

        self.fc1 = nn.Linear(dim_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_actions)

        if continuous_control:
            # trainable parameter
            self._log_std = nn.Parameter(torch.zeros(dim_actions))


    def forward(self, input):
        
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        
        return self.fc3(input)


class Critic(nn.Module):

    def __init__(self, dim_states):
        super(Critic, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> 1
        
        self.fc1 = nn.Linear(dim_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)


    def forward(self, input):
        
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = self.fc3(input)
        
        return input


class ActorCriticAgent:

    def __init__(self, dim_states, dim_actions, actor_lr, critic_lr, gamma, continuous_control=False, gpu = 0):
        
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control
        
        self.device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
        print(f'using {self.device}!')

        self._actor = Actor(self._dim_states, self._dim_actions, self._continuous_control).to(self.device)

        # Adam optimizer
        self._actor_optimizer = Adam(self._actor.parameters(), lr = actor_lr)

        self._critic = Critic(self._dim_states).to(self.device)

        # Adam optimizer
        self._critic_optimizer = Adam(self._critic.parameters(), lr = critic_lr)

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_actor_loss = self._compute_actor_loss_continuous if self._continuous_control else self._compute_actor_loss_discrete


    def select_action(self, observation):
        return self._select_action(observation)
        

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        
        observation = torch.from_numpy(observation).to(self.device)
        
        with torch.no_grad():
            logits = self._actor(observation)
            
        distr = Categorical(logits = logits)
        action = distr.sample().item()
            
        return action

    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter
        
        observation = torch.from_numpy(observation).to(self.device)
        
        with torch.no_grad():
            mu = self._actor(observation)
            std = torch.exp(self._actor._log_std)
        
        distr = Normal(mu, std)
        action = distr.sample().cpu().numpy()
        
        return action


    def _compute_actor_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        
        observation_batch = observation_batch.to(self.device)
        action_batch = torch.from_numpy(action_batch).to(self.device)
        advantage_batch = torch.from_numpy(advantage_batch).to(self.device)
        
        logits = self._actor(observation_batch)
            
        distr = Categorical(logits = logits)
        log_probs = distr.log_prob(action_batch).squeeze().to(self.device)
        
        loss = torch.multiply(-log_probs, advantage_batch)
        
        return torch.mean(loss)


    def _compute_actor_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        
        observation_batch = observation_batch.to(self.device)
        action_batch = torch.from_numpy(action_batch).to(self.device)
        advantage_batch = torch.from_numpy(advantage_batch).to(self.device)
        
        mu = self._actor(observation_batch)
        std = torch.exp(self._actor._log_std)
        
        distr = Normal(mu, std)
        log_probs = distr.log_prob(action_batch).squeeze().to(self.device)
        
        loss = torch.multiply(-log_probs, advantage_batch)
        
        return torch.mean(loss)


    def _compute_critic_loss(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        # minimize mean((r + gamma * V(s_t1) - V(s_t))^2)
        
        observation_batch = torch.from_numpy(observation_batch).to(self.device)
        next_observation_batch = torch.from_numpy(next_observation_batch).to(self.device)
        
        td_estimate = self._critic(observation_batch).squeeze()
        
        #with torch.no_grad():
        #    v_t1 = self._critic(next_observation_batch).squeeze().cpu().numpy()
        #    td_target = reward_batch + self._gamma * v_t1 * (1 - done_batch)
        #    td_target = torch.tensor(td_target, device = self.device).float()
        
        v_t1 = self._critic(next_observation_batch).squeeze()
        td_target = torch.tensor(reward_batch, device = self.device) + self._gamma * v_t1 * torch.tensor(1 - done_batch, device = self.device)
        td_target = td_target.float()
        
        return F.mse_loss(td_estimate, td_target)


    def update_actor(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
        # compute the advantages using the critic and update the actor parameters
        # use self._compute_actor_loss
        
        observation_batch = torch.from_numpy(observation_batch).to(self.device)
        next_observation_batch = torch.from_numpy(next_observation_batch).to(self.device)
        
        with torch.no_grad():
            v_t = self._critic(observation_batch).squeeze().cpu().numpy()
            v_t1 = self._critic(next_observation_batch).squeeze().cpu().numpy()
            advantage_batch = reward_batch + (self._gamma * v_t1) * (1 - done_batch) - v_t
        
        loss = self._compute_actor_loss(observation_batch, action_batch, advantage_batch)
        
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        
        
    def update_critic(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        # update the critic
        # use self._compute_critic_loss
        
        loss = self._compute_critic_loss(observation_batch, reward_batch, next_observation_batch, done_batch)
        
        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()