import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .grad_utils import *

from collections import deque

import logging

class DDPGScheduler(object):
  """DDPG hyperparameter scheduler. It allows to modify hyperparameters at runtime as a function of a global timestep counter.
  At each step it sets the hyperparameter values given by the provided functons

  **Parameters**:
  
  *batch_size_f* (*function*): batch size scheduler function

  *exploration_sdev* (*function*): standard deviation of exploration noise 

  *PER_alpha* (*function*): prioritised experience replay alpha scheduler function

  *PER_beta* (*function*): prioritised experience replay beta scheduler function

  *tau* (*function*): soft target update combination coefficient

  *actor_lr_scheduler_func* (*function*): multiplicative lr update for actor

  *critic_lr_scheduler_func* (*function*): multiplicative lr update for critic

  *steps_per_update* (*function*): number of SGD steps per update

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
    steps_per_update = None):

    self.batch_size_f = batch_size
    self.exploration_sdev_f = exploration_sdev
    self.PER_alpha_f = PER_alpha
    self.tau_f = tau
    self.critic_lr_scheduler_fn = critic_lr_scheduler_fn
    self.actor_lr_scheduler_fn = actor_lr_scheduler_fn
    self.PER_beta_f = PER_beta
    self.steps_per_update_f = steps_per_update if steps_per_update is not None else lambda t: 1

    self.reset()

  def _step(self):

    self.batch_size = self.batch_size_f(self.counter)
    self.exploration_sdev = self.exploration_sdev_f(self.counter)
    self.PER_alpha = self.PER_alpha_f(self.counter)
    self.tau = self.tau_f(self.counter)
    self.PER_beta = self.PER_beta_f(self.counter)
    self.steps_per_update = self.steps_per_update_f(self.counter)

    self.counter += 1

  def reset(self):

    """reset iteration counter
  
    """
    self.counter = 0

    self._step()


class AR1Noise(object):
  """autocrrelated noise process for DDPG exploration

  Parameters:

  *size* (*int*): process sample size

  *seed* (*int*): random seed

  *mu* (*float* or *np.ndarray*): noise mean

  *sigma* (*float* or *np.ndarray*): noise standard deviation
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
    """Reset the internal state."""
    self.state = 0
  #
  def sample(self):
    """Update internal state and return it as a noise sample."""
    self.state = 0.9*self.state + np.random.normal(self.mu,self.sigma,size=1)
    return self.state


class DDPG(object):
  """deterministic deep policy gradient with importance-sampled prioritised experienced replay (PER)

  **Parameters**:

  *actor* (`rlmodels.models.grad_utils.Agent`): model wrapper

  *critic* (`rlmodels.models.grad_utils.Agent`): model wrapper

  *env*: environment object with the same interface as OpenAI gym's environments

  *scheduler* (`DDPGScheduler`): scheduler object that controls hyperparameter values at runtime

  """
  def __init__(self,actor,critic,env,scheduler):

    self.actor = actor
    self.critic = critic
    #self.critic.loss = nn.MSELoss()

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

  def _get_delta(
    self,
    actor,
    critic,
    target_actor,
    target_critic,
    batch,
    discount_rate,
    sample_weights,
    td_steps,
    optimise=True):

    # return delta = PER weights. if optimise = True, agents are updated in-place
    batch_size = len(batch)

    sqrt_sample_weights = (sample_weights**0.5).view(-1,1) # importance sampling weights
    #process batch
    S1 = torch.from_numpy(np.array([x[0] for x in batch])).float()
    #z = np.array([x[1] for x in batch])
    #print(batch[0])
    A1 = torch.from_numpy(np.array([x[1] for x in batch])).float().view(-1,1)
    R = torch.from_numpy(np.array([x[2] for x in batch])).float()
    S2 = torch.from_numpy(np.array([x[3] for x in batch])).float()
    T = torch.from_numpy(np.array([x[4] for x in batch])).float()

    # calculate critic target
    with torch.no_grad():
      A2 = target_actor.forward(S2).view(-1,1)

      Y = R.view(-1,1) + discount_rate**(td_steps)*target_critic.forward(torch.cat((S2,A2),dim=1))*T.view(-1,1) 

    delta = Y - critic.forward(torch.cat((S1,A1),dim=1))

    #optimise critic
    critic.optim.zero_grad()
    if optimise:
      (sample_weights.view(-1,1)*delta**2).mean().backward()
      critic.optim.step()

      if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional debug computations
        with torch.no_grad():
          delta2 = Y - critic.forward(torch.cat((S1,A1),dim=1))
          improvement = torch.abs(delta).mean() - torch.abs(delta2).mean()
        logging.debug("Critic mean loss improvement: {x}".format(x=improvement))

    
    actor.optim.zero_grad()
    # optimise actor
    if optimise:
      q = - torch.mean(critic.forward(torch.cat((S1,actor.forward(S1).view(-1,1)),dim=1)))
      q.backward()
      actor.optim.step()

      if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional debug computations
        with torch.no_grad():
          q2 =- torch.mean(critic.forward(torch.cat((S1,actor.forward(S1).view(-1,1)),dim=1)))
        logging.debug("Actor mean Q-improvement: {x}".format(x=-q2+q))

    return torch.abs(delta).detach().numpy()

  def _step(self,actor, critic, target_actor, target_critic,s1,exploration_sdev,render):
    # perform an action given actor, critic and their targets, the current state, and an epsilon (exploration probability)
    self.noise_process.sigma = exploration_sdev
    with torch.no_grad():
      #eps = torch.from_numpy(np.random.normal(0,exploration_sdev,self.action_dim)).float()
      eps = torch.from_numpy(self.noise_process.sample()).float()
      a = actor.forward(s1) + eps
      a = torch.min(torch.max(a,self.action_low),self.action_high).numpy() #clip action

    sarst = (s1,a) #t = termination

    s2,r,done,info = self.env.step(a)
    if render:
      self.env.render()
    sarst += (float(r),s2,1-int(done)) #t = termination signal (t=0 if s2 is terminal state, t=1 otherwise)

    return sarst

  def _process_td_steps(self,step_list,discount_rate):
    # takes a list of SARST steps as input and outputs an n-steps TD SARST tuple
    s0 = step_list[0][0]
    a = step_list[0][1]
    R = np.sum([step_list[i][2]*discount_rate**(i) for i in range(len(step_list))])
    s1 = step_list[-1][3]
    t = step_list[-1][4]

    return (s0,a,R,s1,t)

  def fit(
    self,
    n_episodes,
    max_ts_by_episode,
    discount_rate=0.99,
    max_memory_size=2000,
    td_steps=1,
    reset=False,
    render=False):

    """
    Fit the agent 
    
    **Parameters**:

    *n_episodes* (*int*): number of episodes to run

    *max_ts_by_episodes* (*int*): maximum number of timesteps to run per episode

    *discount_rate* (*float*): reward discount rate. Defaults to 0.99

    *max_memory_size* (*int*): max memory size for PER. Defaults to 2000

    *td_steps* (*int*): number of temporal difference steps to use in learning

    *reset* (*boolean*): reset trace, scheduler time counter and learning rate time counter to zero if fit has been called before

    *render* (*boolean*): render environment while fitting

    **Returns**:

    (*nn.Module*) updated agent

    """
    logging.info("Running DDPG.fit")
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
    memsize = 0
    step_list = deque(maxlen=td_steps)
    td = 0 #temporal difference step counter

    logging.info("Filling memory...")
    while memsize < max_memory_size:
      # fill step list
      step_list.append(self._step(actor,critic,target_actor,target_critic,s1,scheduler.exploration_sdev,False))
      s1 = step_list[-1][3]
      done = (step_list[-1][4] == 0)
      td += 1
      if td >= td_steps:
        # compute temporal difference n-steps SARST
        td_sarst = self._process_td_steps(step_list,discount_rate)
        memory.add(1,td_sarst)
        memsize +=1

      if done:
        if td < td_steps:
          td_sarst = self._process_td_steps(step_list,discount_rate)
          memory.add(1,td_sarst)
          memsize +=1
        # compute temporal difference n-steps SARST
        td = 0
        step_list = deque(maxlen=td_steps)
        s1 = self.env.reset()

    # fit agents
    logging.info("Training...")
    for i in tqdm(range(n_episodes)):
          
      s1 = self.env.reset()
      self.noise_process.reset()
      ts_reward = 0

      td = 0
      step_list = deque(maxlen=td_steps)

      for j in range(max_ts_by_episode):

        #execute policy
        step_list.append(self._step(actor,critic,target_actor,target_critic,s1,scheduler.exploration_sdev,render))
        td += 1
        s1 = step_list[-1][3]
        r = step_list[-1][2] #latest reward

        if np.isnan(r):
          raise RuntimeError("The model diverged; decreasing step sizes or tau can help to prevent this.")
          
        done = (step_list[-1][4] == 0)

        if td >= td_steps:
          # compute temporal difference n-steps SARST and its delta
          td_sarst = self._process_td_steps(step_list,discount_rate)
          delta = self._get_delta(actor,critic,target_actor,target_critic,[td_sarst],discount_rate,torch.ones(1),td_steps,optimise=False)
          memory.add((delta[0] + 1.0/max_memory_size)**scheduler.PER_alpha,td_sarst)

        # sgd update

        P = memory.total()
        N = memory.get_current_size()
        if N > 0:
          for h in range(scheduler.steps_per_update):
            # get replay batch

            try:
              samples = np.random.uniform(high=P,size=min(scheduler.batch_size,N))
            except OverflowError as e:
              print(e)
              print("it seems that the model parameters are diverging. Decreasing step size or tau might help.")
            
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
            delta = self._get_delta(actor,critic,target_actor,target_critic,batch,discount_rate,batch_w,td_steps,optimise=True)

            #update memory
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
        actor._step()
        critic._step()
        scheduler._step()

        if done:
          # compute temporal difference n-steps SARST and its delta
          break

      if td < td_steps:
        td_sarst = self._process_td_steps(step_list,discount_rate)
        delta = self._get_delta(actor,critic,target_actor,target_critic,[td_sarst],discount_rate,torch.ones(1),td_steps,optimise=False)
        memory.add((delta[0] + 1.0/max_memory_size)**scheduler.PER_alpha,td_sarst)

      self.mean_trace.append(ts_reward)
      logging.info("episode {n}, timestep {ts}, mean reward {x}".format(n=i,x=ts_reward,ts=scheduler.counter))

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
        "mean_reward": self.mean_trace})

    ax = sns.lineplot(data=df,x="episode",y="mean_reward")
    ax.set(xlabel='episode', ylabel='Mean episodic reward')

    plt.show()

  def play(self,n=200):
    """show agent's animation. Only works for OpenAI environments
    
    Parameters:

    *n* (*int*): maximum number of timesteps to visualise. Defaults to 200

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

    *x* (*torch.Tensor*): input vector

    """
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()
    return self.agent.model.forward(x)