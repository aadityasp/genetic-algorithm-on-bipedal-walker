'''
Getting Started (tested on python 3.6.5)

pip install gym
pip install Box2D

https://gym.openai.com/docs/
https://github.com/openai/gym/wiki/BipedalWalker-v2 
(Couldn't find v3 documentation but observations and actions are same as v2)

'''

import gym
import numpy as np
from numpy.random import randn
from numpy.random import rand
import itertools
import random
import pprint as pp

env_name =  "BipedalWalker-v3"
env = gym.make(env_name)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

class HillClimbingAgent():
    def __init__(self, env):
        self.state_dim = 24
        self.build_model()
        
    def build_model(self):
        self.w1 = 1e-4*np.random.rand(self.state_dim, 20)
        self.w2 = 1e-4*np.random.rand(self.state_dim, 20)
        self.w3 = 1e-4*np.random.rand(self.state_dim, 20)
        self.w4 = 1e-4*np.random.rand(self.state_dim, 20)
        self.best_reward = -np.Inf
        self.best_w1 = np.copy(self.w1)
        self.best_w2 = np.copy(self.w2)
        self.best_w3 = np.copy(self.w3)
        self.best_w4 = np.copy(self.w4)
        self.noise_scale = 1e-3
        
    def get_action(self, state):
        p1 = np.dot(state, self.w1)
        p2 = np.dot(state, self.w2)
        p3 = np.dot(state, self.w3)
        p4 = np.dot(state, self.w4)
        action = []
        
        # take neighbor action with maximum weight : argmax
        action.append(self.random_from_action_range(np.argmax(p1)))
        action.append(self.random_from_action_range(np.argmax(p2)))
        action.append(self.random_from_action_range(np.argmax(p3)))
        action.append(self.random_from_action_range(np.argmax(p4)))
        return action
    
    # stochastic hill climbing: take random from nearest best neighbor
    def random_from_action_range(self, index):
        if index == 0:
            return random.uniform(-1, -0.9)
        elif index == 1:
            return random.uniform(-0.9, -0.8)
        elif index == 2:
            return random.uniform(-0.8, -0.7)
        elif index == 3:
            return random.uniform(-0.7, -0.6)
        elif index == 4:
            return random.uniform(-0.6, -0.5)
        elif index == 5:
            return random.uniform(-0.5, -0.4)
        elif index == 6:
            return random.uniform(-0.4, -0.3)
        elif index == 7:
            return random.uniform(-0.4, -0.3)
        elif index == 8:
            return random.uniform(-0.3, -0.2)
        elif index == 9:
            return random.uniform(-0.2, -0.1)
        elif index == 10:
            return random.uniform(-0.1, 0)
        elif index == 11:
            return random.uniform(0, 0.1)
        elif index == 12:
            return random.uniform(0.1, 0.2)
        elif index == 13:
            return random.uniform(0.3, 0.4)
        elif index == 14:
            return random.uniform(0.4, 0.5)
        elif index == 15:
            return random.uniform(0.5, 0.6)
        elif index == 16:
            return random.uniform(0.6, 0.7)
        elif index == 17:
            return random.uniform(0.7, 0.8)
        elif index == 18:
            return random.uniform(0.8, 0.9)
        elif index == 19:
            return random.uniform(0.0, 1)
    
    def update_model(self, reward):
        if reward >= self.best_reward:
            print ("higher reward")
            self.best_reward = reward
            self.best_w1 = np.copy(self.w1)
            self.best_w2 = np.copy(self.w2)
            self.best_w3 = np.copy(self.w3)
            self.best_w4 = np.copy(self.w4)
            self.noise_scale = max(self.noise_scale/2, 1e-3)
        else:
            print ("lower reward")
            self.noise_scale = min(self.noise_scale*2, 2)
            
        # self.w1 = self.best_w1 + self.noise_scale * np.random.rand(self.state_dim, 20)
        # self.w2 = self.best_w2 + self.noise_scale * np.random.rand(self.state_dim, 20)
        # self.w3 = self.best_w3 + self.noise_scale * np.random.rand(self.state_dim, 20)
        # self.w4 = self.best_w4 + self.noise_scale * np.random.rand(self.state_dim, 20)
        self.w1 = self.best_w1 * self.noise_scale
        self.w2 = self.best_w2 * self.noise_scale
        self.w3 = self.best_w3 * self.noise_scale
        self.w4 = self.best_w4 * self.noise_scale


agent = HillClimbingAgent(env)
num_episodes = 1000

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done:
        action = agent.get_action(state)
        #print ("Action: ", action)
        state, reward, done, info = env.step(action)
        if bool(info) == True:
            print ("debug: ", info)
        env.render()
        steps += 1
        total_reward += reward
        
    agent.update_model(total_reward - steps)
    print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))

env.close()