import numpy as np
import torch
import torch.optim as optim
import gym

from rlmodels.models.DQN import *
from rlmodels.nets import VanillaNet

import logging

#logger parameters
FORMAT = '%(asctime)-15s: %(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT,filename="model_fit.log",filemode="a")

max_ep_ts = 200

env = gym.make('CartPole-v0')
env._max_episode_steps = max_ep_ts

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# set hyperparameter runtime schedule as a function of the global number of timesteps
dqn_scheduler = DQNScheduler(
	batch_size = lambda t: 200, #constant
	exploration_rate = lambda t: max(0.01,0.05 - 0.01*int(t/2500)), #decrease exploration down to 1% after 10,000 steps
	PER_alpha = lambda t: 1, #constant
	PER_beta = lambda t: 1, #constant
	tau = lambda t: 100, #constant
	agent_lr_scheduler_fn = lambda t: 1.25**(-int(t/1000)), #decrease step size every 2,500 steps,
	steps_per_update = lambda t: 1) #constant

agent_lr = 0.5 #initial learning rate
agent_model = VanillaNet([60],4,2,None)
agent_opt = optim.SGD(agent_model.parameters(),lr=agent_lr,weight_decay = 0, momentum = 0)

agent = Agent(agent_model,agent_opt)

dqn = DQN(agent,env,dqn_scheduler)

dqn.fit(
	n_episodes=170,
	max_ts_by_episode=max_ep_ts,
	max_memory_size=2000,
	td_steps=1)

dqn.plot()
dqn.play()
