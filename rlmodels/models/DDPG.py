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

import logging

class DDPGScheduler(object):
  """DDPG hyperparameter scheduler. It allows to modify hyperparameters at runtime as a function of a global timestep counter.
  At each step it sets the hyperparameter values given by the provided functons

  Parameters:
  
  `batch_size_f` (`function`): batch size scheduler function

  `exploration_sdev` (`function`): standard deviation of exploration noise 

  `PER_alpha` (`function`): prioritised experience replay alpha scheduler function

  `PER_beta` (`function`): prioritised experience replay beta scheduler function

  `tau` (`function`): soft target update combination coefficient

  `actor_lr_scheduler_func` (`function`): multiplicative lr update for actor

  `critic_lr_scheduler_func` (`function`): multiplicative lr update for critic

  `sgd_update` (`function`): steps between SGD updates as a function of the step counter

  """
  def __init__(
    self,
    batch_size,
    exploration_sdev,
    PER_alpha,
    PER_beta,
    tau,
    actor_lr_scheduler_fn = None,
    critic_lr_scheduler_fn = None,
    sgd_update = None):

    self.batch_size_f = batch_size
    self.exploration_sdev_f = exploration_sdev
    self.PER_alpha_f = PER_alpha
    self.tau_f = tau
    self.critic_lr_scheduler_fn = critic_lr_scheduler_fn
    self.actor_lr_scheduler_fn = actor_lr_scheduler_fn
    self.PER_beta_f = PER_beta
    self.sgd_update_f = sgd_update if sgd_update is None else lambda t: 1

    self.reset()

  def _step(self):

    self.batch_size = self.batch_size_f(self.counter)
    self.exploration_sdev = self.exploration_sdev_f(self.counter)
    self.PER_alpha = self.PER_alpha_f(self.counter)
    self.tau = self.tau_f(self.counter)
    self.PER_beta = self.PER_beta_f(self.counter)
    self.sgd_update = self.sgd_update_f(self.counter)

    self.counter += 1

  def reset(self):

    """reset iteration counter
  
    """
    self.counter = 0

    self._step()


class AR1Noise(object):
  """autocrrelated noise process for DDPG exploration

  Parameters:

  `size` (`int`): process sample size

  `seed` (`int`): random seed

  `mu` (`float` or `np.ndarray`): noise mean

  `sigma` (`float` or `np.ndarray`): noise standard deviation
  """
  #
  def __init__(self, size, seed, mu=0., sigma=0.2):
    """Initialize parameters and noise process."""
    self.mu = mu * np.ones(size)
    self.sigma = sigma
    self.seed = np.random.seed(seed)
    self.reset()
  #
  def reset(self):
    """Reset the internal state (= noise) to mean (mu)."""
    self.state = 0
  #
  def sample(self):
    """Update internal state and return it as a noise sample."""
    self.state = 0.9*self.state + np.random.normal(self.mu,self.sigma,size=1)
    return self.state


class DDPG(object):
  """deterministic deep policy gradient with importance-sampled prioritised experienced replay (PER)

  Parameters:

  `agent` (`rlmodels.models.Agent`): Pytorch neural network model

  `critic` (`rlmodels.models.Agent`): Pytorch neural network model of same class as agent

  `env`: environment object with the same interface as OpenAI gym's environments

  `scheduler` (`DDPGScheduler`): scheduler object that controls hyperparameter values at runtime

  """
  def __init__(self,actor,critic,env,scheduler):

    self.actor = actor
    self.critic = critic
    self.critic.loss = nn.MSELoss()

    self.env = env

    #get input and output dims
    self.state_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.shape[0]

    #get action space boundaries
    self.action_high = torch.from_numpy(env.action_space.high).float()
    self.action_low = torch.from_numpy(env.action_space.low).float()

    self.noise_process = AR1Noise(size=self.action_dim,seed=1)
    self.scheduler = scheduler
    self.mean_trace = []

    if self.scheduler.critic_lr_scheduler_fn is not None:
      self.critic.scheduler = optim.lr_scheduler.LambdaLR(self.critic.optim,self.scheduler.critic_lr_scheduler_fn)

    if self.scheduler.actor_lr_scheduler_fn is not None:
      self.actor.scheduler = optim.lr_scheduler.LambdaLR(self.actor.optim,self.scheduler.actor_lr_scheduler_fn)

  def _update(
    self,
    actor,
    critic,
    target_actor,
    target_critic,
    batch,
    discount_rate, 
    sample_weights):
    # perform gradient descent on actor and critic 

    # return delta = PER weights. agents are updated in-place
    batch_size = len(batch)

    sample_weights = (sample_weights**0.5).view(-1,1)
    #process batch
    S1 = torch.from_numpy(np.array([x[0] for x in batch])).float()
    A1 = torch.from_numpy(np.array([x[1] for x in batch])).float().view(-1,1)
    R = torch.from_numpy(np.array([x[2] for x in batch])).float()
    S2 = torch.from_numpy(np.array([x[3] for x in batch])).float()
    T = torch.from_numpy(np.array([x[4] for x in batch])).float()

    # calculate critic target
    critic.model.zero_grad()
    critic.optim.zero_grad()
    target_critic.model.zero_grad()

    with torch.no_grad():
      A2 = target_actor.forward(S2).view(-1,1)

      Y = R.view(-1,1) + discount_rate*target_critic.forward(torch.cat((S2,A2),dim=1))*T.view(-1,1) 

    Y_hat = critic.forward(torch.cat((S1,A1),dim=1))

    delta = torch.abs(Y_hat-Y).detach().numpy()
    #optimise critic
    critic.loss(sample_weights*Y_hat, sample_weights*Y).backward() #weighted loss
    critic.optim.step()

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional debug computations
      with torch.no_grad():
        Y_hat2 = critic.forward(torch.cat((S1,A1),dim=1))
        delta2 = torch.abs(Y_hat2-Y).detach().numpy()
        improvement = np.mean(delta) - np.mean(delta2)
      logging.debug("Critic mean loss improvement: {x}".format(x=improvement))

    # optimise actor
    actor.model.zero_grad()
    target_actor.model.zero_grad()

    q = - torch.mean(critic.forward(torch.cat((S1,actor.forward(S1).view(-1,1)),dim=1)))
    q.backward()
    actor.optim.step()

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional debug computations
      with torch.no_grad():
        q2 =- torch.mean(critic.forward(torch.cat((S1,actor.forward(S1).view(-1,1)),dim=1)))
      logging.debug("Actor mean Q-improvement: {x}".format(x=-q2+q))

    delta = torch.abs(Y_hat-Y).detach().numpy()
    return delta

  def _step(self,actor, critic, target_actor, target_critic,s1,exploration_sdev,render):
    # perform an action given actor, critic and their targets, the current state, and an epsilon (exploration probability)
    self.noise_process.sigma = exploration_sdev
    with torch.no_grad():
      #eps = torch.from_numpy(np.random.normal(0,exploration_sdev,self.action_dim)).float()
      eps = torch.from_numpy(self.noise_process.sample()).float()
      a = actor.forward(s1) + eps
      a = torch.min(torch.max(a,self.action_low),self.action_high) #clip action

    sarst = (s1,a) #t = termination

    s2,r,done,info = self.env.step(a)
    if render:
      self.env.render()
    sarst += (float(r),s2,1-int(done)) #t = termination signal (t=0 if s2 is terminal state, t=1 otherwise)

    return sarst

  def fit(
    self,
    n_episodes,
    max_ts_by_episode,
    discount_rate=0.99,
    max_memory_size=2000,
    verbose = True,
    reset=False,
    render=False):

    """
    Fit the agent 
    
    Parameters:

    `n_episodes` (`int`): number of episodes to run

    `max_ts_by_episodes` (`int`): maximum number of timesteps to run per episode

    `discount_rate` (`float`): reward discount rate. Defaults to 0.99

    `max_memory_size` (`int`): max memory size for PER. Defaults to 2000

    `verbose` (`boolean`): if true, print mean and max episodic reward each generation. Defaults to True

    `reset_scheduler` (`boolean`): reset trace and scheduler time counter to zero if fit has been called before

    `render` (`boolean`): render environment while fitting

    Returns:
    (`nn.Module`) updated agent

    """

    if reset:
      self.scheduler.reset()
      self.trace = []
      self.critic.scheduler = optim.scheduler.LambdaLR(self.critic.opt,self.scheduler.critic_lr_scheduler_fn)
      self.actor.scheduler = optim.scheduler.LambdaLR(self.actor.opt,self.scheduler.actor_lr_scheduler_fn)

    scheduler = self.scheduler
    actor = self.actor
    critic = self.critic

    # initialize target networks
    target_actor = copy.deepcopy(actor)

    target_critic = copy.deepcopy(critic)

    # initialize and fill memory 
    memory = SumTree(max_memory_size)

    s1 = self.env.reset()
    for i in range(max_memory_size):
      
      sarst = self._step(actor,critic,target_actor,target_critic,s1,scheduler.exploration_sdev,False)
      #print("mem fill sarst: {s}".format(s=sarst))
      memory.add(1,sarst)
      if sarst[4] == 0:
        s1 = self.env.reset()
      else:
        s1 = sarst[3]

    # fit agents
    for i in range(n_episodes):
          
      s1 = self.env.reset()

      ts_reward = 0
      for j in range(max_ts_by_episode):

        #execute epsilon-greedy policy
        sarst = self._step(actor,critic,target_actor,target_critic,s1,scheduler.exploration_sdev,render)
        s1 = sarst[3] #update current state
        r = sarst[2]  #get reward
        done = (sarst[4] == 0) #get termination signal

        #add to memory
        #print("train sarst: {s}".format(s=sarst))
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
            #print("mem data sarst {s}".format(s=sarst))
            batch.append(data) #data from selected leaf
            batch_ids.append(idx)
            batch_p.append(p/P)

          #compute importance sampling weights
          batch_w = np.array(batch_p)
          batch_w = (1.0/(N*batch_w))**scheduler.PER_beta
          batch_w /= np.max(batch_w)
          batch_w = torch.from_numpy(batch_w).float()

          # perform optimisation
          delta = self._update(actor,critic,target_actor,target_critic,batch,discount_rate,batch_w)

          #update memory
          #print(np.mean(delta))
          for k in range(len(delta)):
            memory.update(batch_ids[k],(delta[k] + 1.0/max_memory_size)**scheduler.PER_alpha)

        #target network soft update
        tau = scheduler.tau
        for layer in actor.model._modules:
          target_actor.model._modules[layer].weight.data = (1-tau)*target_actor.model._modules[layer].weight.data + tau*actor.model._modules[layer].weight.data

        for layer in critic.model._modules:
          target_critic.model._modules[layer].weight.data = (1-tau)*target_critic.model._modules[layer].weight.data + tau*critic.model._modules[layer].weight.data

    
        # trace information
        ts_reward += r

        # update learning rate and other hyperparameters
        actor.step()
        critic.step()
        scheduler._step()

        if done:
          self.noise_process.reset()
          break

      self.mean_trace.append(ts_reward/max_ts_by_episode)
      logging.info("episode {n}, timestep {ts}, mean reward {x}".format(n=i,x=ts_reward/max_ts_by_episode,ts=scheduler.counter))

    self.actor = actor
    self.critic = critic

    if render:
      self.env.close()

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
        action = self.actor.model.forward(obs)
        action = torch.min(torch.max(action,self.action_low),self.action_high) #clip action
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
    return self.agent.model.forward(x)