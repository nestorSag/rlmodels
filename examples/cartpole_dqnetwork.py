#run from  project's root
import sys 
import numpy as np

sys.path.insert(0,'models/grad/')
sys.path.insert(0,'nets/')

from DoubleQNetwork import * 
from nets import * 

import torch
import torch.optim as optim

import gym

env = gym.make('CartPole-v0')

agent = VanillaNet([60],4,2,None)
target = VanillaNet([60],4,2,None)

##try DDQNetwork with cartpole

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

ddq = DoubleQNetwork(agent,target,env)


ddq.fit(n_episodes=1000,
	max_ts_by_episode=200,
	batch_size=lambda t: 200,#min(1000,100 + 1000*int(t/2500)),
	exploration_rate_func = lambda t: max(0.01,0.05 - 0.01*int(t/2500)),
	max_memory_size=2000,
	learning_rate=0.001,
	tau=lambda t: 100,#min(500,200 + 100*int(t/2500)),
	scheduler_func=lambda t: 1.25**(-int(t/2500)),
	verbose=True)
