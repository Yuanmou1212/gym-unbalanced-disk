#!/usr/bin/python3

import time
import torch
import torch.nn as nn
import gym, gym_unbalanced_disk, time
import numpy as np
import matplotlib.pyplot as plt
import os

def normalization(theta):
    return (theta+np.pi)%(2*np.pi) - np.pi

class Discretize(gym.Wrapper):
    def __init__(self, env, nvec=10):
        super(Discretize, self).__init__(env) #sets self.env
        self.nvec = nvec

        self.action_space = gym.spaces.Discrete(self.nvec)
        self.alow, self.ahigh = env.action_space.low, env.action_space.high

    def step(self, action):
        action = self.opt_action(action)
        observation, _, done, info = self.env.step(action)
        observation[0] = normalization(observation[0])
        reward = self.get_reward(observation, action=action)

        return np.array(observation), reward, done, info    # Use minus reward here

    def reset(self):
        return np.array(self.env.reset())

    def random_action(self):
        action = self.env.action_space.sample()
        return ((action - self.alow)/(self.ahigh - self.alow)*self.nvec).astype(int)

    def opt_action(self, action):
        action = action / self.nvec *(self.ahigh - self.alow) + self.alow
        return action

    def get_reward(self, observation, action):
        theta = normalization(observation[0])
        omega = observation[1]

        alpha, beta, gamma = 100, 0.05, 0.5
        reward = alpha*theta**2 - beta*omega**2 - gamma*action**2
        return reward



class ActorCritic(nn.Module):
    def __init__(self, env, hidden_size=40):
        super(ActorCritic, self).__init__()
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n

        #define your layers here:
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)  #a)
        self.critic_linear2 = nn.Linear(hidden_size, 1) #a)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size) #a)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions) #a)

    def actor(self, state, return_logp=False):
        #state has shape (Nbatch, Nobs)
        hidden = torch.tanh(self.actor_linear1(state)) #a)
        h = self.actor_linear2(hidden) #a=)
        h = h - torch.max(h,dim=1,keepdim=True)[0] #for additional numerical stability
        logp = h - torch.log(torch.sum(torch.exp(h),dim=1,keepdim=True)) #log of the softmax
        if return_logp:
            return logp
        else:
            return torch.exp(logp) #by default it will return the probability

    def critic(self, state):
        #state has shape (Nbatch, Nobs)
        hidden = torch.tanh(self.critic_linear1(state)) #a)
        return self.critic_linear2(hidden)[:,0] #a) #no activation function

    def forward(self, state):
        #state has shape (Nbatch, Nobs)
        return self.critic(state), self.actor(state)


# def eval_actor(actor_crit, env):
#     pi = lambda x: actor_crit.actor(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
#     with torch.no_grad():
#         rewards_acc = 0
#         obs = env.reset()
#         while True:
#             action = np.argmax(pi(obs)) #b=)
#             obs, reward, done, info = env.step(action)
#             rewards_acc += reward
#             if done:
#                 return rewards_acc

def main():

    # Define environment parameters
    nvec = 10
    max_episode_steps = 1000

    env = gym.make('unbalanced-disk-v0', dt=0.025, umax=3.)
    # env = gym.wrappers.RecordVideo(env, 'A2C', )
    env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=max_episode_steps) #c)
    env = Discretize(env, nvec)

    actor_crit = ActorCritic(env, hidden_size=40)
    # Load for both windows and linus
    file_path = os.path.join('.', 'model', 'A2C_best.pth')
    actor_crit.load_state_dict(torch.load(file_path)) # nvec = 10, hidden_size = 40 'model\A2C_best.pth'  

    traj = []
    omega = []
    Qfun = lambda x: actor_crit.actor(torch.tensor(x[None, :], dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        obs = env.reset()
        try:
            for i in range(1500):
                action = np.argmax(Qfun(obs))
                obs, reward, done, info = env.step(action)
                traj.append(obs[0])
                omega.append(obs[1])
                print(action)
                time.sleep(1/48)
                env.render()
        finally:
            env.close()
            
if __name__ == "__main__":
    main()
