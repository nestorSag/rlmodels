import random
import numpy as np

import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



class CMAES(object):
  """correlation matrix adaptive evolutionary strategy algorithm \n
  
  Parameters: \n
  agent (nn.Module): Pytorch neural network model \n
  env: environment class with roughly the same interface as OpenAI gym's environments, particularly the step() method\n

  """
  
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

  def _calculate_rank(self,vector):
    a={}
    rank=1
    for num in sorted(vector):
      if num not in a:
        a[num]=rank
        rank=rank+1
    return[a[i] for i in vector]

  def fit(self,
      weight_func=None,
      objective_reward = None,
      n_generations=100,
      individuals_by_gen=20,
      episodes_by_ind=10,
      max_ts_by_episode=200,
      alpha_mu= lambda t: 0.99,
      alpha_cm= lambda t: 0.5,
      beta_mu= lambda t: 0,
      beta_cm= lambda t: 0,
      population=None,
      verbose=True):

    """Fit the agent \n
  
    Parameters:\n
    
    weight_func (function): function that maps an individual's ranked performance to recombination weights. defaults to normalised squared ranks (lowest to highest) \n

    objective reward (float): stop when mean episodic reward passes this threshold. Defaults to None\n

    n_generations (int): maximum number of generations to run. Defaults to 100\n

    individuals_by_gen (int): population size for each generation. Defaults to 20\n

    episodes_by_ind (int): how many episodes to run for each individual in the population. Defaults to 10\n

    max_ts_by_episodes (int): maximum number of timesteps to run per episode. Defaults to 200\n

    alpha_mu (float): function that maps generation counts to the step size for the mean vector. Defaults to 0.99 (constant)\n

    alpha_cm (float): function that maps generation counts to the step size for the covariance matrix. Defaults to 0.5 (constant)\n

    beta_mu (float): function that maps generation counts to momentum coefficients for the step of the mean vector. Defaults to 0 (constant)\n

    beta_cm (float): function that maps generation counts to momentum coefficients for the step of covariance matrix. Defaults to 0 (constant)\n

    population (list): initial population for warm start. Defaults to None\n

    verbose (boolean): if true, print mean and max episodic reward each generation. Defaults to True\n

  
    Returns: \n
    (nn.Module) best-performing agent from last generation\n

    """

    #weight_func defaults to normalised squared ranks
    if weight_func is None:
      def weight_func(rewards):
        w = [r**2 for r in self._calculate_rank(rewards)]
        s = sum(w)
        norm_w = [x/s for x in w]
        return norm_w


    #reference architecture structure
    architecture = self.architecture

    if population is None:
      self.mean_trace = []
      self.max_trace = []
      # fill list with randomly generated individuals from initial parameters
      population = self._create_population(individuals_by_gen)
      
    # evaluate population
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
          
          obs = self.env.reset()
          
          for k in range(max_ts_by_episode):
            action = self.agent.forward(obs)
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

  def plot(self):
    """plot mean and max episodic reward for each generation from last fit call\n

    Returns:\n
    list:reward time series\n

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
    """show agent's animation. Only works for OpenAI environments\n
    
    Parameters:\n
    n (int): maximum number of timesteps to visualise. Defaults to 200\n

    """

    obs = self.env.reset()
    for k in range(n):
      action = self.agent.forward(obs)
      obs,reward,done,info = self.env.step(action)
      self.env.render()
      if done:
        break
    self.env.close()

  def forward(self,x):
    """evaluate input with agent\n

    Parameters:\n
    x (torch.Tensor): input vector\n

    """
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()
    return self.agent.forward(x)