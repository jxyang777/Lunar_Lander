import os
import random
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from collections import deque
           
from segment_tree import MinSegmentTree, SumSegmentTree
from ReplayBuffer import PrioritizedReplayBuffer
from Agent import DQNAgent


env_id = "LunarLander-v2"
env = gym.make(env_id)

seed = 777

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)
env.seed(seed)

# parameters
episodes = 1000
memory_size = 10000
batch_size = 64
target_update = 200
epsilon_decay = 0.995

# train
agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)

agent.train(episodes)