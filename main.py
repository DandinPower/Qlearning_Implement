from models.train import DeepQlearning
from models.config import Config
import numpy as np 
import random
import os 
import gym
import time

config = Config()
dpq = DeepQlearning(config)
dpq.LoadParameter()
#dpq.Train()
dpq.play()
#dpq.RandomPlay(30)