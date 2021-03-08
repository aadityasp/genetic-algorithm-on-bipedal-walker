'''
Getting Started (tested on python 3.6.5)

pip install gym
pip install Box2D

https://gym.openai.com/docs/
https://github.com/openai/gym/wiki/BipedalWalker-v2 
(Couldn't find v3 documentation but observations and actions are same as v2)

'''

import gym
env = gym.make('BipedalWalker-v3') #v2 is no longer a valid version
env.reset()
for episode in range(1000):
    env.render()

    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)

    print ("\n\nAction: ", action)
    
    # Action Space: Discrete space allows a fixed range of non-negative numbers, 
    #               so in this case valid actions are either 0 or 1.
    print ("Action space", env.action_space) 
    
    print ("\nObservation: ", observation)
    print ("Total observations: ", len(observation))

    # Observation Space: Box space represents an n-dimensional box, 
    #                    so valid observations will be an array of 4 numbers
    print ("Observation space: ", env.observation_space)
    
    print ("\nReward: ", reward)
    print ("Info: ", info)
    
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

env.close()