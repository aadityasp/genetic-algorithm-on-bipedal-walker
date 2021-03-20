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
import pprint as pp
env = gym.make('BipedalWalker-v3') #v2 is no longer a valid version

for i_episode in range(20):
    stable_walker_states = env.reset()
    for t in range(100):
        env.render()
        #print(stable_walker_states)
        #action = env.action_space.sample()
        #observation, reward, done, info = env.step([1, -1, -1, 1])
        actions = [[1, -1, -1, 1],[-1, 1, 1, -1]]
        if t%2 == 0:
            action = actions[0]
        else:
            action = actions[1]

        observation, reward, done, info = env.step(action)

        # Action Space: Discrete space allows a fixed range of non-negative numbers, 
        #               so in this case valid actions are either 0 or 1.
        # print ("Action space", env.action_space) 
        
        # print ("\nObservation: ", observation)
        # print ("Total observations: ", len(observation))

        # Observation Space: Box space represents an n-dimensional box, 
        #                    so valid observations will be an array of 4 numbers
        # print ("Observation space: ", env.observation_space)
        
        print ("Reward: ", reward, " at step: ", t)
        print (action)
        # print ("Info: ", info)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()