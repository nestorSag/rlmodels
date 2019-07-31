import numpy as np
import torch
import torch.optim as optim
import gym

from rlmodels.models.DDPG import *
from rlmodels.nets import VanillaNet

import logging

#logger parameters
FORMAT = '%(asctime)-15s: %(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT,filename="model_fit.log",filemode="a")

# env parameters
max_ep_ts = 200
env = gym.make('Pendulum-v0')
env._max_episode_steps = max_ep_ts

#reproducibility parameters
env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

#get input and output dims
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
def actor_output(x):
	return 2*torch.tanh(x)

#initial learning rates
actor_lr = 0.0001
critic_lr = 0.001
# set hyperparameter runtime schedule as a function of the global number of timesteps
ddpg_scheduler = DDPGScheduler(
	batch_size = lambda t: 50, #constant
	exploration_sdev = lambda t: 0.3 if t < 20000 else 0.1, #decrease exploration every 10,000 steps
	PER_alpha = lambda t: 1, #constant
	PER_beta = lambda t: 0, #gorw linearly up to 1
	tau = lambda t: 0.001, #constant
	actor_lr_scheduler_fn = lambda t: 1, #multiplicative stepsize update,
	critic_lr_scheduler_fn = lambda t: 1 if t < 15000 else 0.1, #multiplicative stepsize update,
	sgd_update = lambda t: 1) #constant


# create agent models
actor_model = VanillaNet([250,10],input_dim,output_dim,actor_output)
actor_opt = optim.SGD(actor_model.parameters(),lr=actor_lr,weight_decay = 0, momentum = 0)

actor = Agent(
  model = actor_model,
  opt = actor_opt)

critic_model = VanillaNet([250,10],input_dim+output_dim,1,None)
critic_opt = optim.SGD(critic_model.parameters(),lr=critic_lr,weight_decay = 0, momentum = 0)

critic = Agent(
  model = critic_model,
  opt = critic_opt)


ddpg = DDPG(actor,critic,env,ddpg_scheduler)

ddpg.fit(
	n_episodes=200,
	max_ts_by_episode=max_ep_ts,
	max_memory_size=10000,
	render=False)
