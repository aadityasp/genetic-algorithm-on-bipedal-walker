import gym
import numpy as np
from numpy.random import randn
from numpy.random import rand
import itertools
import random
import pprint as pp

env_name = "BipedalWalker-v3"
env = gym.make(env_name)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


class HillClimbingAgent():
    def __init__(self, env):
        self.state_dim = 24
        self.build_model()

    def build_model(self):
        self.w1 = 1e-4 * np.random.rand(self.state_dim, 20)
        self.w2 = 1e-4 * np.random.rand(self.state_dim, 20)
        self.w3 = 1e-4 * np.random.rand(self.state_dim, 20)
        self.w4 = 1e-4 * np.random.rand(self.state_dim, 20)
        self.best_reward = -np.Inf
        self.best_w1 = np.copy(self.w1)
        self.best_w2 = np.copy(self.w2)
        self.best_w3 = np.copy(self.w3)
        self.best_w4 = np.copy(self.w4)
        self.noise_scale = 1e-3
        action_range = []
        # self.action_range_bins = [action_range[i:i + 10] for i in range(0, len(action_range), 10)]
        for i in range(0, 200):
            action_range += [((i / 200.0) * 2) - 1]
        self.action_range_bins = [action_range[i:i + 10] for i in range(0, len(action_range), 10)]

    def get_action(self, state):
        p1 = np.dot(state, self.w1)
        p2 = np.dot(state, self.w2)
        p3 = np.dot(state, self.w3)
        p4 = np.dot(state, self.w4)
        action = [0,0,0,0]


        # final_action=[]
        # take neighbor action with maximum weight : argmax
        # print(action)
        # self.bins(action_range,20)
        # print("action range w1==",action_range_bins[np.argmax(p1)])
        # print("ARGMAX=",np.argmax(p1))
        action[0]=(float(np.random.choice(self.action_range_bins[np.argmax(p1)]))) #self.scale(np.argmax(p1),(0,20),(-1.00,+1.00)))
        action[1]=(float(np.random.choice(self.action_range_bins[np.argmax(p2)]))) #self.scale(np.argmax(p2),(0,20),(-1.00,+1.00)))
        action[2]=(float(np.random.choice(self.action_range_bins[np.argmax(p3)])))#self.scale(np.argmax(p3),(0,20),(-1.00,+1.00)))
        action[3]=(float(np.random.choice(self.action_range_bins[np.argmax(p4)])))#self.scale(np.argmax(p4),(0,20),(-1.00,+1.00)))
        # print(action)
        return action

    # stochastic hill climbing: take random from nearest best neighbor


    def update_model(self, reward):
        if reward >= self.best_reward:
            print("higher reward")
            self.best_reward = reward
            self.best_w1 = np.copy(self.w1)
            self.best_w2 = np.copy(self.w2)
            self.best_w3 = np.copy(self.w3)
            self.best_w4 = np.copy(self.w4)
            self.noise_scale = max(self.noise_scale / 2, 1e-3)
        else:
            print("lower reward")
            self.noise_scale = min(self.noise_scale * 2, 2)
        self.w1 = self.best_w1 + self.noise_scale * np.random.rand(self.state_dim, 20)
        self.w2 = self.best_w2 + self.noise_scale * np.random.rand(self.state_dim, 20)
        self.w3 = self.best_w3 + self.noise_scale * np.random.rand(self.state_dim, 20)
        self.w4 = self.best_w4 + self.noise_scale * np.random.rand(self.state_dim, 20)

    def retain_weights(self, good,weights_best):
        print("Inside retain weights function")

        if good:
            self.w1,self.w2,self.w3,self.w4 = weights_best
            print("better reward, retaining weights")
            self.best_w1 = np.copy(self.w1)
            self.best_w2 = np.copy(self.w2)
            self.best_w3 = np.copy(self.w3)
            self.best_w4 = np.copy(self.w4)
            self.noise_scale = max(self.noise_scale / 2, 1e-3)
        else:
            print("lower reward")
            self.noise_scale = min(self.noise_scale * 2, 2)
        self.w1 = self.best_w1 + self.noise_scale * np.random.rand(self.state_dim, 20)
        self.w2 = self.best_w2 + self.noise_scale * np.random.rand(self.state_dim, 20)
        self.w3 = self.best_w3 + self.noise_scale * np.random.rand(self.state_dim, 20)
        self.w4 = self.best_w4 + self.noise_scale * np.random.rand(self.state_dim, 20)

    def current_best_weights(self):
        return [self.w1, self.w2, self.w3, self.w4]
agent = HillClimbingAgent(env)
num_episodes = 1000
reward_list = []
weights_lst = []
for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:

        action = agent.get_action(state)
        # print ("Action: ", action)
        state, reward, done, info = env.step(action)
        if bool(info) == True:
            print("debug: ", info)
        # print("reward, steps=",reward,steps)
        env.render()
        steps += 1
        total_reward += (reward)

        # total_reward -= steps

    if len(reward_list)>2 :
        # print("reward, steps=",reward,steps)
        if total_reward - 0.1 *steps >= min(reward_list):
            print("Inside good reward")
            print(reward_list, total_reward)
            if len(weights_lst) > 2:
                weights_bst= weights_lst[np.argmin(reward_list)]
                # print("Weights_bestt=",weights_bst)
                # print("Weights_listt=", weights_lst)
                agent.retain_weights(True,weights_bst)
        else:
            print(reward_list, total_reward)
            agent.retain_weights(False,weights_lst[-1])
    reward_list += [total_reward]
    weights_lst += [agent.current_best_weights()]
    # agent.update_model(total_reward - steps)
    print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))
