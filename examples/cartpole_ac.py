import numpy as np 
import torch
import torch.optim as optim
import gym

from rlmodels.models.AC import *
from rlmodels.nets import VanillaNet, DiscretePolicy

import logging

FORMAT = '%(asctime)-15s: %(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT,filename="model_fit.log",filemode="a")

max_ep_ts = 500
env = gym.make('CartPole-v0')
env._max_episode_steps = max_ep_ts

input_dim = env.observation_space.shape[0]
output_dim = 2

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

ac_scheduler = ACScheduler(
  actor_lr_scheduler_fn = lambda t: 1, #reduce step size after some time (multiplicative update)
  critic_lr_scheduler_fn = lambda t: 1) #reduce step size after some time (multiplicative update)

actor_lr = 1e-3
actor_model = DiscretePolicy(layer_sizes=[100,100],input_size=input_dim,output_size=output_dim)
actor_opt = optim.SGD(actor_model.parameters(),lr=actor_lr,weight_decay = 0, momentum = 0)

actor = Agent(
  model = actor_model,
  opt = actor_opt)

critic_lr = 1e-2
critic_model = VanillaNet([100,100],input_dim,1,None)
critic_opt = optim.SGD(critic_model.parameters(),lr=critic_lr,weight_decay = 0, momentum = 0)

critic = Agent(
  model = critic_model,
  opt = critic_opt)


ac = AC(actor,critic,env,ac_scheduler)


ac.fit(
  n_episodes=2500,
  tmax=np.Inf, #episodic updates
  max_ts_by_episode=max_ep_ts)

ac.plot()
ac.play(n=max_ep_ts)