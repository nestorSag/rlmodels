import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
  
from collections import deque   

"""
provides efficient memory data sctructure (fast retrieves and updates).
source of the SumTree class code : https://github.com/jaromiru/AI-blog/blob/master/SumTree.py

@param capacity: number of tree leaves
"""
class SumTree:
  write = 0
  current_size=0

  def __init__(self, capacity):
    # INPUT
    # capacity: number of tree leaves
    self.capacity = capacity
    self.tree = np.zeros( 2*capacity - 1 )
    self.data = np.zeros( capacity, dtype=object )

  def _propagate(self, idx, change):
    parent = (idx - 1) // 2

    self.tree[parent] += change

    if parent != 0:
      self._propagate(parent, change)

  def _retrieve(self, idx, s):
    left = 2 * idx + 1
    right = left + 1

    if left >= len(self.tree):
      return idx

    if s <= self.tree[left]:
      return self._retrieve(left, s)
    else:
      return self._retrieve(right, s-self.tree[left])

  def get_current_size(self):
    return self.current_size

  """
  returns the sum of leaf weights
  """
  def total(self):
    return self.tree[0]

  """
    adds data to tree, potetntially overwritting older data

    @param p: leaf weight
    @param data: leaf data
  """
  def add(self, p, data):
  
    idx = self.write + self.capacity - 1

    self.data[self.write] = data
    self.update(idx, p)

    self.write += 1
    if self.write >= self.capacity:
      self.write = 0

    self.current_size = min(self.current_size+1,self.capacity)

  """
  updates leaf weight

  @param idx: leaf index
  @param p: new weight
  """
  def update(self, idx, p):
    change = p - self.tree[idx]

    self.tree[idx] = p
    self._propagate(idx, change)

  """
  get leaf corresponding to numeric value

  @param s: numeric value
  @return leaf id, tree node id, leaf data
  """
  def get(self, s):

    idx = self._retrieve(0, s)
    dataIdx = idx - self.capacity + 1

    return (idx, self.tree[idx], self.data[dataIdx])


"""
neural network gradient optimisation wrapper

@param model: Pytorch neural network model 
@param optim_: Pytorch optimizer object 
@param loss: pytorch loss function
@param scheduler_func: Python learning rate scheduler
"""
class Agent(object):
  
  def __init__(self,model,optim_,loss,scheduler_func):
    self.model = model
    self.optim = optim_
    self.loss = loss

    self.optim.zero_grad()
    self.scheduler = optim.lr_scheduler.LambdaLR(self.optim,lr_lambda=[scheduler_func])

  def forward(self,x):
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()
    return self.model.forward(x)

"""
double Q network with prioritised experienced replay (PER)

@param agent: Pytorch neural network model
@param target: Pytorch neural network model of same class as agent
@ param env: environment object with the same interface as OpenAI gym's environments
"""

class DoubleQNetwork(object):

  def __init__(self,agent,target,env):

    self.agent = agent
    self.target = target
    self.env = env

    self.mean_trace = []

  def _update(self,agent1,agent2,batch,discount_rate):
    # perform gradient descent on agent1 
    # delta = PER weights
    batch_size = len(batch)

    #process batch
    try:
      S1 = torch.from_numpy(np.array([x[0] for x in batch])).float()
      A1 = torch.from_numpy(np.array([x[1] for x in batch])).long()
      R = torch.from_numpy(np.array([x[2] for x in batch])).float()
      S2 = torch.from_numpy(np.array([x[3] for x in batch])).float()
      T = torch.from_numpy(np.array([x[4] for x in batch])).float()

      _, A2 = torch.max(agent1.forward(S2).detach(),1) #decide with target network

      Y = R.view(-1,1) + discount_rate*agent2.forward(S2).detach().gather(1,A2.view(-1,1))*T.view(-1,1) #evaluate with q network

      Y_hat = agent1.forward(S1).gather(1,A1.view(-1,1)) #optimise target network
    
    #optimise
      agent1.loss(Y_hat,Y).backward()
      agent1.optim.step()

      delta = torch.abs(Y_hat-Y).detach().numpy()

      agent1.optim.zero_grad()

      return delta

    except:
      print(batch) 

  def _step(self,agent,target,s1,eps):
    # perform an action given an agent, a target, the current state, and an epsilon (exploration probability)
    q = agent.forward(s1).detach()

    if np.random.binomial(1,eps,1)[0]:
      a = np.random.randint(low=0,high=list(q.shape)[0],size=1)[0]
    else:
      _, a = q.max(0) #argmax
      a = int(a.numpy())

    sarst = (s1,a) #t = termination

    s2,r,done,info = self.env.step(a)
    sarst += (r,s2,1-int(done)) #t = termination signal (t=0 if s2 is terminal state, t=1 otherwise)

    return sarst


    
  """
  Fit the agent 

  @param n_episodes: number of episodes to run
  @param max_ts_by_episodes: maximum number of timesteps to run per episode
  @param batch_size: function that maps a global timestep counter to a batch size. Defaults to 100 (constant)
  @param exploration_rate_func: function that maps a global timestep counter to an exploration rate. Defaults to 0.05 (constant)
  @param PER_alpha_func: function that maps a global timestep counter to a PER alpha parameter. Defaults to 1 (constant)
  @param tau: function that maps a golbal timestep counter to a target network hard update time window. Defaults to 200 (constant)
  @param learning_rate: SGD learning rate. Defaults to 0.001
  @param discount_rate: reward discount rate. Defaults to 0.99
  @param max_memory_size: max memory size for PER. Defaults to 2000
  @param scheduler_func: function that maps a global timestep counter to a learning rate multiplicative update (for Pytorch LambdaLR scheduler). Defaults to None
  @param verbose: if true, print mean and max episodic reward each generation. Defaults to True
  @return updated agent
  """
  def fit(
    self,
    n_episodes,
    max_ts_by_episode,
    batch_size= lambda t: 100,
    exploration_rate_func = lambda t: 0.05,
    PER_alpha_func = lambda t: 1,
    tau= lambda t: 200,
    learning_rate = 0.001,
    discount_rate=0.99,
    max_memory_size=2000,
    scheduler_func=None,
    verbose = True):

    if scheduler_func is None:
    	scheduler_func = lambda t: 1

    def weights_init(m):
      torch.nn.init.xavier_uniform(m.weight.data)

    # initialize agents
    agent = Agent(
      model = self.agent,
      optim_ = optim.SGD(self.agent.parameters(),lr=learning_rate,weight_decay = 0, momentum = 0),
      loss = nn.MSELoss(),
      scheduler_func = scheduler_func)

    target = Agent(
      model = self.target,
      optim_ = optim.SGD(self.agent.parameters(),lr=learning_rate,weight_decay = 0, momentum = 0),
      loss = nn.MSELoss(),
      scheduler_func = scheduler_func)

    # initialize and fill memory 
    memory = SumTree(max_memory_size)

    s1 = self.env.reset()
    for i in range(max_memory_size):
    	
    	sarst = self._step(agent,target,s1,exploration_rate_func(0))
    	memory.add(1,sarst)
    	if sarst[4] == 0:
    		s1 = self.env.reset()
    	else:
    		s1 = sarst[3]

    # fit the agent
    global_step_counter = 0
    for i in range(n_episodes):
          
      s1 = self.env.reset()

      ts_reward = 0
      for j in range(max_ts_by_episode):

        #execute epsilon-greedy policy
        sarst = self._step(agent,target,s1,exploration_rate_func(global_step_counter))

        s1 = sarst[3] #update current state

        r = sarst[2]
        done = sarst[4] == 0
        #get batch
        M = memory.total()

        samples = np.random.uniform(high=M,size=min(batch_size(global_step_counter)-1,memory.get_current_size()))
        batch = [] #always use latest sample
        batch_ids = []
        for u in samples:
          idx, _ ,data = memory.get(u)
          batch.append(data) #data from selected leaf
          batch_ids.append(idx)

        batch += [sarst]

        # update lr
        agent.scheduler.step()
        # perform optimisation
        delta = self._update(agent,target,batch,discount_rate)

        if (global_step_counter % tau(global_step_counter)) == (tau(global_step_counter)-1):
          target.model.load_state_dict(agent.model.state_dict())

        #update memory
        sarst = batch.pop() #take out latest sample from batch (it's not in memory yet)
        for k in range(len(batch)):
          memory.update(batch_ids[k],delta[k]**PER_alpha_func(global_step_counter))

        memory.add(delta[-1]**PER_alpha_func(global_step_counter),sarst)

        # trace information
        ts_reward += r

        global_step_counter += 1
        if done:
          break

      self.mean_trace.append(ts_reward/max_ts_by_episode)
      if verbose:
      	print("episode {n}, mean reward {x}".format(n=i,x=ts_reward/max_ts_by_episode))

    self.agent = agent.model
    self.target = target.model

    return agent.model, target.model

  """
  plot mean reward from last fit call

  @return reward time series
  """
  def plot(self):

    if len(self.mean_trace)==0:
      print("The trace is empty.")
    else:
      df = pd.DataFrame({
        "episode":list(range(len(self.mean_trace))),
        "mean reward": self.mean_trace})

    sns.lineplot(data=df,x="episode",y="mean reward")
    plt.show()

  """
  show agent's animation. Only works for OpenAI environments

  @param n: number of timesteps to visualise. Defaults to 500
  """
  def play(self,n=500):
    obs = self.env.reset()
    for k in range(n):
      action = np.argmax(self.agent.forward(obs).detach().numpy())
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