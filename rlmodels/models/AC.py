import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .grad_utils import *

import logging

from collections import deque

class ACScheduler(object):
  """AC hyperparameter scheduler. It allows to modify hyperparameters at runtime as a function of a global timestep counter.
  At each step it sets the hyperparameter values given by the provided functons

  **Parameters**:
  
  *actor_lr_scheduler_func* (*function*): multiplicative lr update for actor

  *critic_lr_scheduler_func* (*function*): multiplicative lr update for critic

  """
  def __init__(
    self,
    actor_lr_scheduler_fn = None,
    critic_lr_scheduler_fn = None):

    self.critic_lr_scheduler_fn = critic_lr_scheduler_fn
    self.actor_lr_scheduler_fn = actor_lr_scheduler_fn

    self.reset()

  def _step(self):

    self.counter += 1

  def reset(self):

    """reset iteration counter
  
    """
    self.counter = 0

    self._step()


class AC(object):
  """actor-critic policy gradient algorithm

  **Parameters**:

  *actor* ( `rlmodels.models.grad_utils.Agent` ): model wrapper. Its *model* attribute must be a network that returns a suitable *torch.distributions* object from which to sample actions

  *critic* (`rlmodels.models.grad_utils.Agent`): model wrapper

  *env*: environment object with the same interface as OpenAI gym's environments

  *scheduler* (`ACScheduler`): scheduler object that controls hyperparameter values at runtime

  """
  def __init__(self,actor,critic,env,scheduler):

    self.actor = actor
    self.critic = critic
    self.critic.loss = nn.MSELoss()

    self.env = env

    if len(self.env.action_space.shape) == 0:
      self.action_is_discrete = True
    else:
      self.action_is_discrete = False
      #get action space boundaries
      self.action_high = torch.from_numpy(env.action_space.high).float()
      self.action_low = torch.from_numpy(env.action_space.low).float()


    self.scheduler = scheduler
    self.mean_trace = []

    if self.scheduler.critic_lr_scheduler_fn is not None:
      self.critic.scheduler = optim.lr_scheduler.LambdaLR(self.critic.optim,self.scheduler.critic_lr_scheduler_fn)

    if self.scheduler.actor_lr_scheduler_fn is not None:
      self.actor.scheduler = optim.lr_scheduler.LambdaLR(self.actor.optim,self.scheduler.actor_lr_scheduler_fn)

    actor_npars = sum(p.numel() for p in actor.model.parameters() if p.requires_grad)
    critic_npars = sum(p.numel() for p in critic.model.parameters() if p.requires_grad)

    #bound gradients in an attempt to prevent divergence
    #assuming a change of 0.1 on average per parameter is ok (pre learning rate)
    self.actor_gnb = np.sqrt(actor_npars*1e-1) #actor gradient norm bound
    self.critic_gnb = np.sqrt(critic_npars*1e-1) #critic gradient norm bound

  def _update(
    self,
    actor,
    critic,
    batch,
    discount_rate,
    optimise=True):

    # return delta = PER weights. if optimise = True, agents are updated in-place
    batch_size = len(batch)

    #process batch
    S1 = torch.from_numpy(np.array([x[0] for x in batch])).float()
    A1 = torch.from_numpy(np.array([x[1] for x in batch])).view(-1,1)
    R = torch.from_numpy(np.array([x[2] for x in batch])).float()
    S2 = torch.from_numpy(np.array([x[3] for x in batch])).float()
    T = torch.from_numpy(np.array([x[4] for x in batch])).float()

    time_idx = np.array(range(batch_size))
    step_discounts = torch.from_numpy(discount_rate**((1+time_idx)[::-1])).float().view(-1,1)

    with torch.no_grad():
      Y = R.view(-1,1) + step_discounts*critic.forward(S2)*T.view(-1,1) #evaluate with target network

    delta = Y - critic.forward(S1)

    critic.optim.zero_grad()
    #optimise critic
    if optimise:
      (delta**2).mean().backward() #weighted loss
      torch.nn.utils.clip_grad_norm_(critic.model.parameters(), self.critic_gnb) #clip gradient
      critic.optim.step()

      if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional debug computations
        with torch.no_grad():
          delta2 = Y - critic.forward(S1)
          improvement = np.mean(torch.abs(delta).detach().numpy()) - np.mean(torch.abs(delta2).detach().numpy())
        logging.debug("Critic mean loss improvement: {x}".format(x=improvement))

    actor.optim.zero_grad()
    delta = delta.detach()
    # optimise actor
    if optimise:
      pi = actor.forward(S1)
      pg = - (pi.log_prob(A1)*delta).mean()
      pg.backward()
      torch.nn.utils.clip_grad_norm_(actor.model.parameters(), self.actor_gnb) #clip gradient; normal policies tend to have exploding gradients
      actor.optim.step()

      if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional debug computations
        with torch.no_grad():
          pi = actor.forward(S1)
          pg2 = - (pi.log_prob(A1)*(delta.detach())).mean()
          logging.debug("pi: {x}".format(x=pi))
          logging.debug("action: {x}".format(x=A1))
          logging.debug("weighted log density change: {x}".format(x=pg-pg2))

  def _step(self,actor, critic,s1,render):
    # perform an action given an actor and critic, the current state, and an epsilon (exploration probability)
    with torch.no_grad():
      dist = actor.forward(s1)
      #print(dist.probs)
      try:
        a = dist.sample()
      except RuntimeError as e:
        raise RuntimeError("The model diverged; decreasing step sizes or increasing the minimum allowed variance of continuous policies can help to prevent this.")
        print(e)

      if self.action_is_discrete:
        a = int(a)
      else:
        a = torch.min(torch.max(a,self.action_low),self.action_high) #clip action

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
    tmax = 1,
    reset=False,
    render=False):

    """
    Fit the agent 
    
    **Parameters**:

    *n_episodes* (int): number of episodes to run

    *max_ts_by_episodes* (*int*): maximum number of timesteps to run per episode

    *discount_rate* (*float*): reward discount rate. Defaults to 0.99

    *tmax* (*int*): number of timesteps between k-step TD updates, k=1,...,tmax. An infinite value implies episodic updates

    *reset* (*boolean*): reset trace, scheduler time counter and learning rate time counter to zero if fit has been called before

    *render* (*boolean*): render environment while fitting

    **Returns**:

    (nn.Module) updated agent

    """
    logging.info("Running AC.fit")
    if reset:
      self.scheduler.reset()
      self.trace = []
      self.critic.scheduler = optim.scheduler.LambdaLR(self.critic.opt,self.scheduler.critic_lr_scheduler_fn)
      self.actor.scheduler = optim.scheduler.LambdaLR(self.actor.opt,self.scheduler.actor_lr_scheduler_fn)

    scheduler = self.scheduler
    actor = self.actor
    critic = self.critic

    # fit agents
    logging.info("Training...")

    for i in range(n_episodes):

      ts_reward = 0

      step_list = deque(maxlen=tmax if isinstance(tmax,int) else None)
      td = 0

      s1 = self.env.reset()

      for j in range(max_ts_by_episode):

        #execute policy
        step_list.append(self._step(actor,critic,s1,render))
        td += 1

        s1 = step_list[-1][3]
        r = step_list[-1][2] #latest reward

        if np.isnan(r):
          raise RuntimeError("The model diverged; decreasing step sizes or increasing the minimum allowed variance of continuous policies can help to prevent this.")
        
        done = (step_list[-1][4] == 0)

        if td == tmax:
          processed = [self._process_td_steps([step_list[j] for j in range(i,tmax)],discount_rate) for i in range(tmax)]
          self._update(actor,critic,processed,discount_rate,optimise=True)
          td = 0

        # trace information
        ts_reward += r

        # update learning rate and other hyperparameters
        actor._step()
        critic._step()
        scheduler._step()

        if done:

          break

      if td > 0:
        processed = [self._process_td_steps([step_list[-td+j] for j in range(i,td)],discount_rate) for i in range(td)]
        self._update(actor,critic,processed,discount_rate,optimise=True)

      self.mean_trace.append(ts_reward)
      logging.info("episode {n}, timestep {ts}, mean reward {x}".format(n=i,x=ts_reward,ts=scheduler.counter))

    self.actor = actor
    self.critic = critic

    if render:
      self.env.close()

  def plot(self):
    """plot mean timestep reward from last `fit` call

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
    
    **Parameters**:

    *n* (*int*): maximum number of timesteps to visualise. Defaults to 200

    """
    with torch.no_grad():
      obs = self.env.reset()
      for k in range(n):
        pi = self.actor.model.forward(obs)
        action = pi.sample()
        if self.action_is_discrete:
          action = int(action)
        obs,reward,done,info = self.env.step(action)
        self.env.render()
        if done:
          break
      self.env.close()

  def forward(self,x):
    """evaluate input with agent

    **Parameters**:

    *x* (*torch.Tensor*): input vector

    """
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()
    return self.agent.model.forward(x)