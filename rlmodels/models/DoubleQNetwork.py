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

from .grad_utils import *

class DoubleQNetworkScheduler(object):
  """double Q network hyperparameter scheduler. It allows to modify hyperparameters at runtime as a function of a global timestep counter.
  At each step it sets the hyperparameter values given by the provided functons

  Parameters:
  
  `batch_size_f` (`function`): batch size scheduler function

  `exploration_rate` (`function`): exploration rate scheduler function 

  `PER_alpha` (`function`): prioritised experience replay alpha scheduler function

  `PER_beta` (`function`): prioritised experience replay beta scheduler function

  `tau` (`function`): hard target update time window scheduler function

  `learning_rate_update` (`function`): multiplicative update factor scheduler function to be passed to torch LambdaLR scheduler

  `sgd_update` (`function`): steps between SGD updates as a function of the step counter

  """
  def __init__(
    self,
    batch_size,
    exploration_rate,
    PER_alpha,
    PER_beta,
    tau,
    learning_rate_update,
    sgd_update):

    self.batch_size_f = batch_size
    self.exploration_rate_f = exploration_rate
    self.PER_alpha_f = PER_alpha
    self.tau_f = tau
    self.PER_beta_f = PER_beta
    self.learning_rate_f = learning_rate_update
    self.sgd_update_f = sgd_update

    self.reset()

  def _step(self):

    self.batch_size = self.batch_size_f(self.counter)
    self.exploration_rate = self.exploration_rate_f(self.counter)
    self.PER_alpha = self.PER_alpha_f(self.counter)
    self.tau = self.tau_f(self.counter)
    self.PER_beta = self.PER_beta_f(self.counter)
    self.learning_rate = self.learning_rate_f(self.counter)
    self.sgd_update = self.sgd_update_f(self.counter)

    self.counter += 1

  def reset(self):

    """reset iteration counter
  
    """
    self.counter = 0

    self._step()

class DoubleQNetwork(object):
  """double Q network with importance-sampled prioritised experienced replay (PER)

  Parameters:

  `agent` (`torch.nn.Module`): Pytorch neural network model

  `target` (`torch.nn.Module`): Pytorch neural network model of same class as agent

  `env`: environment object with the same interface as OpenAI gym's environments

  `scheduler` (`DoubleQNetworkScheduler`): scheduler object that controls hyperparameter values at runtime

  """
  def __init__(self,agent,env,scheduler):

    self.agent = agent
    self.env = env
    self.scheduler = scheduler
    self.mean_trace = []

  def _update(self,agent1,agent2,batch,discount_rate, sample_weights):
    # perform gradient descent on agent1 

    # return delta = PER weights. agents are updated in-place
    batch_size = len(batch)

    sample_weights = (sample_weights**0.5).view(-1,1)
    #process batch
    S1 = torch.from_numpy(np.array([x[0] for x in batch])).float()
    A1 = torch.from_numpy(np.array([x[1] for x in batch])).long()
    R = torch.from_numpy(np.array([x[2] for x in batch])).float()
    S2 = torch.from_numpy(np.array([x[3] for x in batch])).float()
    T = torch.from_numpy(np.array([x[4] for x in batch])).float()

    with torch.no_grad():
      _, A2 = torch.max(agent1.forward(S2),1) #decide with q network

      Y = R.view(-1,1) + discount_rate*agent2.forward(S2).gather(1,A2.view(-1,1))*T.view(-1,1) #evaluate with target network

    Y_hat = agent1.forward(S1).gather(1,A1.view(-1,1)) #optimise q network
  
  #optimise
    agent1.loss(sample_weights*Y_hat, sample_weights*Y).backward() #weighted loss
    agent1.optim.step()

    delta = torch.abs(Y_hat-Y).detach().numpy()

    agent1.optim.zero_grad()

    return delta

  def _step(self,agent,target,s1,eps):
    # perform an action given an agent, a target, the current state, and an epsilon (exploration probability)
    with torch.no_grad():
      q = agent.forward(s1)

    if np.random.binomial(1,eps,1)[0]:
      a = np.random.randint(low=0,high=list(q.shape)[0],size=1)[0]
    else:
      _, a = q.max(0) #argmax
      a = int(a.numpy())

    sarst = (s1,a) #t = termination

    s2,r,done,info = self.env.step(a)
    sarst += (r,s2,1-int(done)) #t = termination signal (t=0 if s2 is terminal state, t=1 otherwise)

    return sarst

  def fit(
    self,
    n_episodes,
    max_ts_by_episode,
    initial_learning_rate=0.001,
    discount_rate=0.99,
    max_memory_size=2000,
    verbose = True,
    reset=False):

    """
    Fit the agent 
    
    Parameters:

    `n_episodes` (`int`): number of episodes to run

    `max_ts_by_episodes` (`int`): maximum number of timesteps to run per episode

    `initial_learning_rate` (`float`): initial SGD learning rate

    `discount_rate` (`float`): reward discount rate. Defaults to 0.99

    `max_memory_size` (`int`): max memory size for PER. Defaults to 2000

    `verbose` (`boolean`): if true, print mean and max episodic reward each generation. Defaults to True

    `reset_scheduler` (`boolean`): reset trace and scheduler time counter to zero if fit has been called before

    Returns:
    (`nn.Module`) updated agent

    """
    if reset:
      self.scheduler.reset()
      self.trace = []

    scheduler = self.scheduler

    # initialize agents
    agent = Agent(
      model = self.agent,
      optim_ = optim.SGD(self.agent.parameters(),lr=initial_learning_rate,weight_decay = 0, momentum = 0),
      loss = nn.MSELoss(),
      scheduler_func = scheduler.learning_rate_f)

    target = Agent(
      model = copy.deepcopy(self.agent),
      optim_ = optim.SGD(self.agent.parameters(),lr=initial_learning_rate,weight_decay = 0, momentum = 0),
      loss = nn.MSELoss(),
      scheduler_func = scheduler.learning_rate_f)

    # initialize and fill memory 
    memory = SumTree(max_memory_size)

    s1 = self.env.reset()
    for i in range(max_memory_size):
    	
    	sarst = self._step(agent,target,s1,scheduler.exploration_rate)
    	memory.add(1,sarst)
    	if sarst[4] == 0:
    		s1 = self.env.reset()
    	else:
    		s1 = sarst[3]

    # fit agent
    for i in range(n_episodes):
          
      s1 = self.env.reset()

      ts_reward = 0
      for j in range(max_ts_by_episode):

        #execute epsilon-greedy policy
        sarst = self._step(agent,target,s1,scheduler.exploration_rate)

        s1 = sarst[3] #update current state
        r = sarst[2]  #get reward
        done = (sarst[4] == 0) #get termination signal

        #add to memory
        memory.add(1,sarst) #default initial weight of 1

        # sgd update
        if scheduler.counter % scheduler.sgd_update ==0:
          # get replay batch
          P = memory.total()
          N = memory.get_current_size()

          samples = np.random.uniform(high=P,size=min(scheduler.batch_size,N))
          batch = []
          batch_ids = []
          batch_p = []
          for u in samples:
            idx, p ,data = memory.get(u)
            batch.append(data) #data from selected leaf
            batch_ids.append(idx)
            batch_p.append(p/P)

          #compute importance sampling weights
          batch_w = np.array(batch_p)
          batch_w = (1.0/(N*batch_w))**scheduler.PER_beta
          batch_w /= np.max(batch_w)
          batch_w = torch.from_numpy(batch_w).float()

          # perform optimisation
          delta = self._update(agent,target,batch,discount_rate,batch_w)

          #update memory
          for k in range(len(delta)):
            memory.update(batch_ids[k],(delta[k] + 1.0/max_memory_size)**scheduler.PER_alpha)

          #target network hard update
        if (scheduler.counter % scheduler.tau) == (scheduler.tau-1):
          target.model.load_state_dict(agent.model.state_dict())

    
        # trace information
        ts_reward += r

        # update learning rate and other hyperparameters
        agent.scheduler.step()
        scheduler._step()

        if done:
          break

      self.mean_trace.append(ts_reward/max_ts_by_episode)
      if verbose:
      	print("episode {n}, timestep {ts}, mean reward {x}".format(n=i,x=ts_reward/max_ts_by_episode,ts=scheduler.counter))

    self.agent = agent.model
    self.target = target.model

    return agent.model, target.model

  def plot(self):
    """plot mean episodic reward from last `fit` call

    """

    if len(self.mean_trace)==0:
      print("The trace is empty.")
    else:
      df = pd.DataFrame({
        "episode":list(range(len(self.mean_trace))),
        "mean reward": self.mean_trace})

    sns.lineplot(data=df,x="episode",y="mean reward")
    plt.show()

  def play(self,n=200):
    """show agent's animation. Only works for OpenAI environments
    
    Parameters:

    `n` (`int`): maximum number of timesteps to visualise. Defaults to 200

    """
    with torch.no_grad():
      obs = self.env.reset()
      for k in range(n):
        action = np.argmax(self.agent.forward(obs).numpy())
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