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

class REINFORCEScheduler(object):
  """REINFORCE hyperparameter scheduler. It allows to modify hyperparameters at runtime as a function of a global timestep counter.
  At each step it sets the hyperparameter values given by the provided functons

  Parameters:
  
  `batch_size_f` (`function`): batch size scheduler function

  `exploration_sdev` (`function`): standard deviation of exploration noise 

  `PER_alpha` (`function`): prioritised experience replay alpha scheduler function

  `PER_beta` (`function`): prioritised experience replay beta scheduler function

  `tau` (`function`): soft target update combination coefficient

  `actor_lr_scheduler_func` (`function`): multiplicative lr update for actor

  `critic_lr_scheduler_func` (`function`): multiplicative lr update for critic

  `steps_per_update` (`function`): number of SGD steps per update

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


class REINFORCE(object):
  """actor critic REINFORCE episodic algorithm

  Parameters:

  `actor` (`rlmodels.models.Agent`): Pytorch neural network model

  `critic` (`rlmodels.models.Agent`): Pytorch neural network model of same class as agent

  `env`: environment object with the same interface as OpenAI gym's environments

  `scheduler` (`DDPGScheduler`): scheduler object that controls hyperparameter values at runtime

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
    batch,
    discount_rate,
    optimise=True):

    # return delta = PER weights. if optimise = True, agents are updated in-place
    batch_size = len(batch)

    #process batch
    S1 = torch.from_numpy(np.array([x[0] for x in batch])).float()
    A1 = torch.from_numpy(np.array([x[1] for x in batch])).view(-1,1)
    #R = torch.from_numpy(np.array([x[2] for x in batch])).float()
    S2 = torch.from_numpy(np.array([x[3] for x in batch])).float()
    T = torch.from_numpy(np.array([x[4] for x in batch])).float()

    #format discounted rewards per timestep
    time_idx = np.array(range(batch_size))
    R = np.array([x[2] for x in batch])
    R = R*discount_rate**(time_idx)
    R = np.cumsum(R[::-1])[::-1]
    R = R/(discount_rate**(time_idx))
    R = torch.from_numpy(R).float()

    critic.optim.zero_grad()

    delta = R.view(-1,1) - critic.forward(S1)

    #optimise critic
    if optimise:
      (delta**2).mean().backward() #weighted loss
      critic.optim.step()

      if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional debug computations
        with torch.no_grad():
          delta2 = R.view(-1,1) - critic.forward(S1)
          improvement = np.mean(delta) - np.mean(delta2)
        logging.debug("Critic mean loss improvement: {x}".format(x=improvement))

    actor.optim.zero_grad()
    delta = delta.detach()
    # optimise actor
    if optimise:
      pi = actor.forward(S1)
      #A1 = pi.sample()
      #pg = - (pi.log_prob(A1)*advantage*sample_weights.view(-1,1)).mean()
      pg = - (pi.log_prob(A1)*delta).mean()
      pg.backward()
      actor.optim.step()

      if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional debug computations
        with torch.no_grad():
          pi = actor.forward(S1)
          pg2 = - (pi.log_prob(A1)*advantage).mean()
          logging.debug("pi: {x}".format(x=pi.probs))
          logging.debug("action: {x}".format(x=A1))
          logging.debug("weighted log density change: {x}".format(x=pg-pg2))

  def _step(self,actor, critic,s1,render):
    # perform an action given actor, critic and their targets, the current state, and an epsilon (exploration probability)
    with torch.no_grad():
      dist = actor.forward(s1)
      #print(dist.probs)
      a = dist.sample()
      if self.action_is_discrete:
        a = int(a)

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
    verbose = True,
    reset=False,
    render=False):

    """
    Fit the agent 
    
    Parameters:

    `n_episodes` (`int`): number of episodes to run

    `max_ts_by_episodes` (`int`): maximum number of timesteps to run per episode

    `discount_rate` (`float`): reward discount rate. Defaults to 0.99

    `reset` (`boolean`): reset trace, scheduler time counter and learning rate time counter to zero if fit has been called before

    `render` (`boolean`): render environment while fitting

    Returns:
    (`nn.Module`) updated agent

    """
    logging.info("Running REINFORCE.fit")
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

      step_list = deque(maxlen=None)
      s1 = self.env.reset()

      for j in range(max_ts_by_episode):

        #execute policy
        step_list.append(self._step(actor,critic,s1,render))
        s1 = step_list[-1][3]
        r = step_list[-1][2] #latest reward
        done = (step_list[-1][4] == 0)

        # trace information
        ts_reward += r

        # update learning rate and other hyperparameters
        actor.step()
        critic.step()
        scheduler._step()

        if done:
          break

      # perform optimisation
      self._update(actor,critic,step_list,discount_rate,optimise=True)

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

    Parameters:

    `x` (`torch.Tensor`): input vector

    """
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()
    return self.agent.model.forward(x)