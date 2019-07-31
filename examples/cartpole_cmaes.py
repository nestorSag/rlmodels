import numpy as np 
import torch
import torch.optim as optim
import gym

from rlmodels.models.CMAES import *
from rlmodels.nets import VanillaNet

FORMAT = '%(asctime)-15s: %(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT,filename="model_fit.log",filemode="a")

env = gym.make('CartPole-v0')

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# agent output is argmax from a 2-dimensional output vector (values not related to Q function!) 
def binary_output(x):
  return np.argmax(x.detach().numpy())

#CMAES is not gradient-based, so we don't have to wrap the model in an Agent instance
agent = VanillaNet([6,6],4,2,binary_output)

# set hyperparameter runtime schedule as a function of the global number of timesteps
# set them to constants for this example
cmaes_scheduler = CMAESScheduler(
  alpha_mu = lambda t: 0.99,
  alpha_cm = lambda t: 0.5,
  beta_mu = lambda t: 0,
  beta_cm = lambda t: 0) #constant

cmaes = CMAES(agent,env,cmaes_scheduler)

# in CMAES we want new populations to be centered at some weighted combination of the best-performing agents from the previous population
# the weight function determines how much each individual agent influences the next sample
# more weight should be assigned to agents that have higher rewards

## in this example, rewards are ranked and maped as rank ==> rank**4 and normalised internally
def wf(ranks):
  return ranks**4

cmaes.fit(weight_func=wf,
      reward_objective = None,
      n_generations=20,
      individuals_by_gen=20,
      episodes_by_ind=30,
      max_ts_by_episode=200,
      verbose=True)
