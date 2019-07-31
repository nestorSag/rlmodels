import random
import numpy as np

import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

class CMAESScheduler(object):
  """CMAES hyperparameter scheduler. It allows to modify hyperparameters at runtime as a function of a global generation counter.
  At each generation it sets the hyperparameter values given by the provided functons

  Parameters:
  
  `alpha_mu` (`function`): step size scheduler for the mean parameter 

  `alpha_cm` (`function`): step size scheduler for the covariance matrix parameter

  `beta_mu` (`function`): momentum term for the mean vector parameter

  `beta_cm` (`function`): momentum term for covariance matrix parameter

  """
  def __init__(
    self,
    alpha_mu,
    alpha_cm,
    beta_mu,
    beta_cm):

    self.alpha_mu_f = alpha_mu
    self.alpha_cm_f = alpha_cm
    self.beta_mu_f = beta_mu
    self.beta_cm_f = beta_cm

    self.reset()

  def _step(self):

    self.alpha_mu = self.alpha_mu_f(self.counter)
    self.alpha_cm = self.alpha_cm_f(self.counter)
    self.beta_mu = self.beta_mu_f(self.counter)
    self.beta_cm = self.beta_cm_f(self.counter)

    self.counter += 1

  def reset(self):

    """reset iteration counter
  
    """
    self.counter = 0

    self._step()


class CMAES(object):
  """correlation matrix adaptive evolutionary strategy algorithm 
  
  Parameters: 

  `agent` (`torch.nn.Module`): Pytorch neural network model 

  `env`: environment class with roughly the same interface as OpenAI gym's environments, particularly the step() method

  `scheduler` (`CMAESScheduler`): scheduler object that controls hyperparameter values at runtime

  """
  
  def __init__(self,agent,env,scheduler):

    self.agent = agent
    # reference architecture architecture
    self.architecture = self.agent.state_dict()

    # get parameter space dimensionality
    d = 0
    for layer in self.architecture:
      d += np.prod(self.architecture[layer].shape)
    self.d = d

    self.env = env
    self.scheduler = scheduler
    self.max_trace = []
    self.mean_trace = []

    #initialise mean and covariance matrix
    self.mu = torch.from_numpy(np.zeros((self.d,1))).float()
    self.cm = torch.from_numpy(np.eye(self.d)).float()

    #initialise mean and covariance momentum terms
    self.update_mu = torch.from_numpy(np.zeros((self.d,1))).float()
    self.update_cm = torch.from_numpy(np.zeros(self.d,self.d)).float()

  def _unroll_params(self,population):
    # unroll neural architecture weights into a long vector
    # OUTPUT
    # matrix whose columns are the population parameter vectors
    unrolled_matrix = torch.empty(self.d,0)
    for ind in population:
      architecture = ind["architecture"]
      unrolled = torch.empty(0,1)
      for layer in architecture:
        unrolled = torch.cat([unrolled,architecture[layer].view(-1,1)],0)
      unrolled_matrix = torch.cat([unrolled_matrix,unrolled],1)

    return unrolled_matrix

  def _get_population_statistics(self,population):
    # OUTPUT
    # weighted population mean
    # aggregated rank 1 updates for covariance matrix

    n = len(population)
    unrolled_matrix = self._unroll_params(population)
    weights = torch.from_numpy(np.array([ind["weight"] for ind in population]).reshape(-1,1)).float()
    
    # compute weighted mean as a matrix vector product
    w_mean = torch.mm(unrolled_matrix,weights)

    m_y, n_y = unrolled_matrix.shape

    y = (unrolled_matrix - self.mu)

    r1updates = torch.zeros(m_y,m_y)

    for i in range(n_y):
      col = y[:,i]
      r1updates += weights[i]*torch.ger(col,col) 

    return w_mean, r1updates

  def _roll(self,unrolled):
    # roll a long vector into the agent's structure
    architecture = self.architecture
    rolled = {}
    s0=0
    for layer in architecture:
      if len(architecture[layer].shape) == 2:
        m,n = architecture[layer].shape
        rolled[layer] = unrolled[s0:(s0+m*n)].view(m,n)
      else:
        m = architecture[layer].shape[0]
        n = 1
        rolled[layer] = unrolled[s0:(s0+m*n)].view(m)
      
      s0 += m*n
    return rolled 

  def _create_population(self,n):
    population = []
    for i in range(n):
      eps = np.random.multivariate_normal(self.mu.numpy()[:,0],self.cm.numpy(),1)
      torch_eps = torch.from_numpy(eps).float().view(self.d,1)
      ind_architecture = self._roll(torch_eps)
      population.append({
          "architecture":ind_architecture,
          "avg_episode_r":0})

    return population

  def _calculate_rank(self,vector):
    # calculate vector ranks from lowest(1) to highest (len(vector))

    a={}
    rank=1
    for num in sorted(vector):
      if num not in a:
        a[num]=rank
        rank=rank+1
    return np.array([a[i] for i in vector])

  def fit(self,
      weight_func=None,
      reward_objective = None,
      n_generations=100,
      individuals_by_gen=20,
      episodes_by_ind=10,
      max_ts_by_episode=200,
      verbose=True,
      reset=False,
      debug_logger=False):

    """Fit the agent 
  
    Parameters:
    
    `weight_func` (`function`): function that maps individual ranked (lowest to highest) performances to (normalised to sum 1) recombination weights. It has to work on `numpy` arrays; defaults to quadratic function

    `objective reward` (`float`): stop when max episodic reward passes this threshold. Defaults to `None`

    `n_generations` (`int`): maximum number of generations to run. Defaults to 100

    `individuals_by_gen` (`int`): population size for each generation. Defaults to 20

    `episodes_by_ind` (`int`): how many episodes to run for each individual in the population. Defaults to 10

    `max_ts_by_episodes` (`int`): maximum number of timesteps to run per episode. Defaults to 200

    `verbose` (`boolean`): if true, print mean and max episodic reward each generation. Defaults to True

    `reset` (`boolean`): reset scheduler counter to zero and performance traces if `fit` has been called before

  
    Returns: 
    (`torch nn.Module`) best-performing agent from last generation

    """

    if debug_logger:
      logging.basicConfig(level=logging.DEBUG)
    else:
      logging.basicConfig(level=logging.INFO)


    if reset:
      self.scheduler.reset()
      self.mean_trace = []
      self.max_trace = []
    #weight_func defaults to normalised squared ranks

    scheduler = self.scheduler

    if weight_func is None:
      def weight_func(ranks):
        return ranks**2


    #reference architecture structure
    architecture = self.architecture

    population = self._create_population(individuals_by_gen)
      
    # evaluate population
    i = 0
    reward_objective = np.Inf if reward_objective is None else reward_objective
    best = -np.Inf

    while i < n_generations and best < reward_objective:

      for l in range(len(population)):
        # set up nn agent
        agent = population[l]

        self.agent.load_state_dict(agent["architecture"])

        #interact with environment
        for j in range(episodes_by_ind):
          
          ep_reward = 0 
          
          obs = self.env.reset()
          
          for k in range(max_ts_by_episode):
            with torch.no_grad():
              action = self.agent.forward(obs)
            obs,reward,done,info = self.env.step(action)
            
            ep_reward += reward/max_ts_by_episode #avg intra episode reward

            if done:
              break

          population[l]["avg_episode_r"] += ep_reward/episodes_by_ind #avg reward

      # calculate weights for each individual
      population_rewards = np.array([ind["avg_episode_r"] for ind in population])
      weights = weight_func(self._calculate_rank(population_rewards))

      if ((np.argsort(population_rewards) - np.argsort(weights)) != 0).any():
        logging.warning("Warning: recombination weights function does not preserve rank order")

      norm_weights = weights/np.sum(weights)

      #print(population_rewards)
      #print(norm_weights)

      for k in range(len(population)):
        population[k]["weight"] = norm_weights[k]

      #debug info
      self.mean_trace.append(np.mean(population_rewards))
      self.max_trace.append(np.max(population_rewards))
      logging.info("generation {n}, mean trace {x}, max trace {y}".format(n=i,x=np.mean(population_rewards),y=np.max(population_rewards)))

      w_mean, r1updates = self._get_population_statistics(population)

      #update gradient with momentum
      self.update_cm = scheduler.beta_cm*self.update_cm + r1updates - self.cm
      self.update_mu = scheduler.beta_mu*self.update_mu + w_mean - self.mu

      #update parameters
      self.cm = self.cm + scheduler.alpha_cm*self.update_cm
      self.mu = self.mu + scheduler.alpha_mu*self.update_mu

      # update agent to the best performing one in current population
      self.agent.load_state_dict(population[np.argmax(norm_weights)]["architecture"])

      population = self._create_population(individuals_by_gen)
      i += 1
      best = np.max(population_rewards) # best avg episodic reward 

      scheduler._step()

    return self.agent

  def plot(self):
    """plot mean and max episodic reward for each generation from last fit call

    """
    if len(self.mean_trace)==0:
      print("The traces are empty.")
    else:
      df = pd.DataFrame({
        "generation":list(range(len(self.max_trace))) + list(range(len(self.max_trace))),
        "value": self.max_trace + self.mean_trace,
        "trace": ["max" for x in self.max_trace] + ["mean" for x in self.mean_trace]})

      sns.lineplot(data=df,x="generation",y="value",hue="trace")
      plt.show()

  def play(self,n=200):
    """show agent's animation. Only works for OpenAI environments
    
    Parameters:

    `n` (`int`): maximum number of timesteps to visualise. Defaults to 200

    """

    obs = self.env.reset()
    with torch.no_grad():
      for k in range(n):
        action = self.agent.forward(obs)
        obs,reward,done,info = self.env.step(action)
        self.env.render()
        if done:
          break
      self.env.close()

  def forward(self,x):
    """evaluate input with agent

    Parameters:

    `x` (`torch.Tensor`): input vector

    """
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()
    return self.agent.forward(x)