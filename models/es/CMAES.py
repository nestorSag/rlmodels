import random
import numpy as np

import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
correlation matrix adaptive evolutionary strategy algorithm

@param agent: Pytorch neural network model 
@param env: environment class with roughly the same interface as OpenAI gym's environments, particularly the step() method
"""

class CMAES(object):
  
  def __init__(self,agent,env):

    self.agent = agent
    # reference architecture architecture
    self.architecture = self.agent.state_dict()

    # get parameter space dimensionality
    d = 0
    for layer in self.architecture:
      d += np.prod(self.architecture[layer].shape)
    self.d = d

    self.env = env
    self.last_trace = []

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

  """
  Fit the agent
  @param weight_func: function that maps an individual's ranked performance to recombination weights
  @param objective reward: stop when mean episodic reward passes this threshold. Defaults to None
  @param n_generations: maximum number of generations to run. Defaults to 100
  @param individuals_by_gen: population size for each generation. Defaults to 20
  @param episodes_by_ind: how many episodes to run for each individual in the population. Defaults to 10
  @param max_ts_by_episodes: maximum number of timesteps to run per episode. Defaults to 200
  @param alpha_mu: function that maps generation counts to the step size for the mean vector. Defaults to 0.99 (constant)
  @param alpha_cm: function that maps generation counts to the step size for the covariance matrix. Defaults to 0.5 (constant)
  @param beta_mu: function that maps generation counts to momentum coefficients for the step of the mean vector. Defaults to 0 (constant)
  @param beta_cm: function that maps generation counts to momentum coefficients for the step of covariance matrix. Defaults to 0 (constant)
  @param seed: numpy and environment random seed. Defaults to 1
  @param population: initial population for warm start. Defaults to None
  @param verbose: if true, print mean and max episodic reward each generation. Defaults to True
  @return best-performing agent from last generation
  """

  def fit(self,
      weight_func,
      objective_reward = None,
      n_generations=100,
      individuals_by_gen=20,
      episodes_by_ind=10,
      max_ts_by_episode=200,
      alpha_mu= lambda t: 0.99,
      alpha_cm= lambda t: 0.5,
      beta_mu= lambda t: 0,
      beta_cm= lambda t: 0,
      seed=1,
      population=None,
      verbose=True):

    #reference architecture structure
    architecture = self.architecture
    np.random.seed(seed)

    if population is None:
      self.mean_trace = []
      self.max_trace = []
      # fill list with randomly generated individuals from initial parameters
      population = self._create_population(individuals_by_gen)
      
    # evaluate population
    self.env.seed(seed)
    i = 0
    objective_reward = np.Inf if objective_reward is None else objective_reward
    best = -np.Inf

    while i < n_generations and best < objective_reward:

      for l in range(len(population)):
        # set up nn agent
        agent = population[l]

        self.agent.load_state_dict(agent["architecture"])

        #interact with environment
        for j in range(episodes_by_ind):
          
          ep_reward = 0 
          
          #self.env.seed(seed)
          obs = self.env.reset()
          
          for k in range(max_ts_by_episode):
            action = self.agent.forward(obs).detach().numpy()
            obs,reward,done,info = self.env.step(action)
            
            ep_reward += reward/max_ts_by_episode #avg intra episode reward

            if done:
              break

          population[l]["avg_episode_r"] += ep_reward/episodes_by_ind #avg reward

      # calculate weights for each individual
      population_rewards = [ind["avg_episode_r"] for ind in population]
      ind_weights = weight_func(population_rewards)

      #print(population_rewards)
      #print(ind_weights)

      for k in range(len(population)):
        population[k]["weight"] = ind_weights[k]

      #debug info
      self.mean_trace.append(np.mean(population_rewards))
      self.max_trace.append(np.max(population_rewards))
      if verbose:
        print("generation {n}, mean trace {x}, max trace {y}".format(n=i,x=np.mean(population_rewards),y=np.max(population_rewards)))

      w_mean, r1updates = self._get_population_statistics(population)

      self.update_cm = beta_cm(i)*self.update_cm + r1updates - self.cm
      self.update_mu = beta_mu(i)*self.update_mu + w_mean - self.mu

      self.cm = self.cm + alpha_cm(i)*self.update_cm
      self.mu = self.mu + alpha_mu(i)*self.update_mu

      #self.cm = (1 - alpha_cm)*self.cm + alpha_cm*r1updates
      #self.mu = (1 - alpha_mu)*self.mu + alpha_mu*w_mean

      # update agent to the best performing one in current population
      self.agent.load_state_dict(population[np.argmax(ind_weights)]["architecture"])

      population = self._create_population(individuals_by_gen)
      i += 1
      best = np.max(population_rewards) # best avg episodic reward 
      # update population

    return self.agent

  """
  plot mean and max episodic reward for each generation from last fit call

  @return reward time series
  """
  def plot(self):
    if len(self.mean_trace)==0:
      print("The traces are empty.")
    else:
      df = pd.DataFrame({
        "generation":list(range(len(self.max_trace))) + list(range(len(self.max_trace))),
        "value": self.max_trace + self.mean_trace,
        "type": ["min" for x in self.max_trace] + ["mean" for x in self.mean_trace]})

      sns.lineplot(data=df,x="generation",y="value",hue="type")
      plt.show()

  """
  show agent's animation. Only works for OpenAI environments

  @param n: number of timesteps to visualise. Defaults to 500
  """
  def play(self,n=500):

    obs = self.env.reset()
    for k in range(n):
      action = self.agent.forward(obs).detach.numpy()
      obs,reward,done,info = self.env.step(action)
      self.env.render()
    self.env.close()

  """
  evaluate input with agent

  @param x: input vector
  """
  def forward(self,x):
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()
    return self.agent.forward(x)