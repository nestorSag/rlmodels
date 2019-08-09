import numpy as np 
import torch
import torch.optim as optim
import gym

from rlmodels.models.DQN import *
from rlmodels.nets import VanillaNet

import logging

FORMAT = '%(asctime)-15s: %(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT,filename="model_fit.log",filemode="a")

env = gym.make('LunarLander-v2')
max_ep_ts = 1000

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# set hyperparameter runtime schedule as a function of the global number of timesteps
dqn_scheduler = DQNScheduler(
  batch_size = lambda t: 200 if t < 75000 else 300, #constant
  exploration_rate = lambda t: 0.05 if t < 75000 else 0.01, #decrease exploration down to 1% after 10,000 steps
  PER_alpha = lambda t: 1, #constant
  PER_beta = lambda t: 1, #constant
  tau = lambda t: 100, #constant
  agent_lr_scheduler_fn = lambda t: 1 if t < 75000 else 0.1, #decrease step size every 2,500 steps,
  steps_per_update = lambda t: 1) #constant
  
agent_lr = 0.01 #initial learning rate
agent_model = VanillaNet([200,200],8,4,None)
agent_opt = optim.SGD(agent_model.parameters(),lr=agent_lr,weight_decay = 0, momentum = 0)

agent = Agent(agent_model,agent_opt)

dqn = DQN(agent,env,dqn_scheduler)

dqn.fit(
  n_episodes=350,
  max_ts_by_episode=max_ep_ts,
  max_memory_size=2000,
  td_steps=5)

dqn.plot()
dqn.play()