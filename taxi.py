import os 
import gym
import random
import numpy as np 
import time
from models.train import DeepQlearning

'''
state : 500種
action : 6種
'''
env = gym.make('Taxi-v3')
env._max_episode_steps = 5000000
dpq = DeepQlearning(env, _stateNum = 500, _actionNum = 6, _hiddenSize = 50, _batchSize = 16)
dpq.Train(200)
dpq.play()