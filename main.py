import os 
import gym
import random
import numpy as np 
import time
from models.train import DeepQlearning
from models.config import Config
'''
state : 500種
action : 6種
'''
config = Config()
dpq = DeepQlearning(config)
#dpq.LoadParameter()
dpq.Train()
#dpq.play()