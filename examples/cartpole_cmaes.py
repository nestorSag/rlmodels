import numpy as np 
import torch
import torch.optim as optim
import gym

from rlmodels.models.es import CMAES
from rlmodels.nets import VanillaNet

env = gym.make('CartPole-v0')

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# agent output is argmax from a 2-dimensional output vector (values not related to Q function!) 
def binary_output(x):
  return np.argmax(x.detach().numpy())

agent = VanillaNet([6,6],4,2,binary_output)
cmaes = CMAES(agent,env)

# in CMAES we want new populations to be centered at some weighted combination of the best-performing agents from the previous population
# the weight function determines how much each individual agent influences the next sample
# more weight should be assigned to agents that have higher rewards

## in this example, rewards are ranked and maped as rank ==> rank**4/sum(rank**4 for r in ranks)
def wf(rewards_list):
  def calculate_rank(vector):
    a={}
    rank=1
    for num in sorted(vector):
      if num not in a:
        a[num]=rank
        rank=rank+1
    return[a[i] for i in vector]
  #
  #
  ranks = [r**4 for r in calculate_rank(rewards_list)]
  s = sum(ranks)
  ranks = [r/s for r in ranks]
  return ranks

cmaes.fit(weight_func=wf,
      objective_reward = None,
      n_generations=20,
      individuals_by_gen=20,
      episodes_by_ind=30,
      max_ts_by_episode=200,
      alpha_mu= lambda t: 0.99,
      alpha_cm= lambda t: 0.5,
      beta_mu= lambda t: 0,
      beta_cm= lambda t: 0,
      population=None,
      verbose=True)
