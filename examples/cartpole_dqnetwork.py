import numpy as np

from rlmodels.models.grad import DoubleQNetwork
from rlmodels.nets import VanillaNet

import torch
import torch.optim as optim

import gym

env = gym.make('CartPole-v0')

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

agent = VanillaNet([60],4,2,None)
target = VanillaNet([60],4,2,None)


ddq = DoubleQNetwork(agent,target,env)


ddq.fit(n_episodes=1000,
	max_ts_by_episode=200,
	batch_size=lambda t: 200,
	exploration_rate_func = lambda t: max(0.01,0.05 - 0.01*int(t/2500)), #decrease exploration down to 1% after 10,000 steps
	max_memory_size=2000,
	learning_rate=0.001,
	tau=lambda t: 100,
	scheduler_func=lambda t: 1.25**(-int(t/2500)), #decrease step size a bit every 2,500 steps
	verbose=True)
