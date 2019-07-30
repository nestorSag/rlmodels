import numpy as np
import torch
import torch.optim as optim
import gym

from rlmodels.models.DDPG import *
from rlmodels.nets import VanillaNet

env = gym.make('Pendulum-v0')

#get input and output dims
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

def actor_output(x):
	return 2*torch.tanh(x)

actor = VanillaNet([32,64],input_dim,output_dim,actor_output)

critic = VanillaNet([50,50],input_dim+output_dim,1,None)

# set hyperparameter runtime schedule as a function of the global number of timesteps
ddpg_scheduler = DDPGScheduler(
	batch_size = lambda t: 256, #constant
	exploration_sdev = lambda t: 0.4**(1+int(t/1000)), #decrease exploration every 10,000 steps
	PER_alpha = lambda t: 1, #constant
	PER_beta = lambda t: 1, #gorw linearly up to 1
	tau = lambda t: 0.001, #constant
	actor_learning_rate_update = lambda t: 1, #decrease step size every 2,500 steps,
	critic_learning_rate_update = lambda t: 1, #decrease step size every 2,500 steps,
	sgd_update = lambda t: 1) #constant

ddpg = DDPG(actor,critic,env,ddpg_scheduler)

ddpg.fit(
	n_episodes=100,
	max_ts_by_episode=200,
	actor_initial_learning_rate=0.0001,
	critic_initial_learning_rate=0.001,
	max_memory_size=10000,
	verbose=True)