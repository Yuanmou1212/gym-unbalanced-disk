import gym, gym_unbalanced_disk, time
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, A2C, SAC
#from UnbalancedDisk import *
from gym_unbalanced_disk import UnbalancedDisk
from gym_unbalanced_disk import *

def normalization(theta):
    return (theta+np.pi)%(2*np.pi) - np.pi

def get_diff(theta,target):
    return 2*np.pi-np.abs(theta-target) if np.abs(theta-target)>np.pi else np.abs(theta-target)

class UnbalancedDisk_multi(UnbalancedDisk):
    """limit theta to [-pi,pi]"""
    # observation [theta,omega,target]
    def __init__(self, umax=3., dt = 0.025):
        super(UnbalancedDisk_multi, self).__init__(umax=umax, dt=dt)
        low = [-np.pi,-40.,-np.pi]
        high = [np.pi,40.,np.pi]
        weight_omega, weight_action = 0.15, 1e-3
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(3,))
        self.reward_fun = lambda self: (-150*(get_diff(normalization(self.target),normalization(self.th)))**2 - weight_omega *self.omega **2 - weight_action*self.u**2).item()
        self.count = 0

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        self.count += 1
        iteration = (self.count//200)%4 # every 200 interation change the target
        if iteration%2 == 0:
            self.target = np.pi
        elif iteration == 1:
            self.target = np.pi - np.radians(10)
        else:
            self.target = np.pi + np.radians(10)
        self.target = normalization(self.target)
        self.theta = normalization(self.th_noise)
        return np.array([self.theta, self.omega_noise, self.target]) #change anything here

    def reset(self):
        self.th = np.random.normal(loc=0, scale=0.001)
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.u = 0
        self.count = -1
        return self.get_obs()



max_episode_steps = 5000
env = UnbalancedDisk_multi(dt=0.025, umax=3.)
env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=max_episode_steps)

# model = SAC.load('model/SAC_multi_best.zip')

model = SAC("MlpPolicy", env, learning_rate=0.01, verbose=1)
model.learn(total_timesteps=5000)
model.save('model/SAC_Multi_5000')


obs = env.reset()
angle_ = []
target_ = []
try:
    for i in range(5000):
        iteration = (i//200)%3
        if iteration == 0:
            target = np.pi
        elif iteration == 1:
            target = np.pi - np.radians(10)
        else: 
            target = np.pi + np.radians(10)
        obs[2] = normalization(target)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        print(f'target = {obs[2]}, theta = {obs[0]}')
        time.sleep(1/120)
        angle_.append(normalization(obs[0]))
        target_.append(obs[2])

        if done:
            env.reset()

finally:
    env.close()


# angle_ = np.array(angle_)
# target_ = np.array(target_)
# save_dic = dict(angle_=angle_)
# np.savez('model/angle.npz', **save_dic)
# save_dic = dict(target_=target_)
# np.savez('model/target.npz', **save_dic)