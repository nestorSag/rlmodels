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

from collections import deque

from .grad_utils import *

import logging

class DQNScheduler(object):
  """double Q network hyperparameter scheduler. It allows to modify hyperparameters at runtime as a function of a global timestep counter.
  At each step it sets the hyperparameter values given by the provided functons

  **Parameters**:
  
  *batch_size_f* (*function*): batch size scheduler function

  *exploration_rate* (*function*): exploration rate scheduler function 

  *PER_alpha* (*function*): prioritised experience replay alpha scheduler function

  *PER_beta* (*function*): prioritised experience replay beta scheduler function

  *tau* (*function*): hard target update time window scheduler function

  *agent_lr_scheduler_fn* (*function*): multiplicative lr update for actor

  *steps_per_update* (*function*): number of SGD steps per update

  """
  def __init__(
    self,
    batch_size,
    exploration_rate,
    PER_alpha,
    PER_beta,
    tau,
    agent_lr_scheduler_fn=None,
    steps_per_update=None):

    self.batch_size_f = batch_size
    self.exploration_rate_f = exploration_rate
    self.PER_alpha_f = PER_alpha
    self.tau_f = tau
    self.PER_beta_f = PER_beta
    self.steps_per_update_f = steps_per_update if steps_per_update is not None else lambda t: 1

    self.agent_lr_scheduler_fn = agent_lr_scheduler_fn

    self.reset()

  def _step(self):

    self.batch_size = self.batch_size_f(self.counter)
    self.exploration_rate = self.exploration_rate_f(self.counter)
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

class DQN(object):
  """double Q network with importance-sampled prioritised experienced replay (PER)

  **Parameters**:

  *agent* (`rlmodels.models.grad_utils.Agent`): model wrapper

  *env*: environment object with the same interface as OpenAI gym's environments

  *scheduler* (`DQNScheduler`): scheduler object that controls hyperparameter values at runtime

  """
  def __init__(self,agent,env,scheduler):

    self.agent = agent
    self.env = env
    self.scheduler = scheduler
    self.mean_trace = []

    if self.scheduler.agent_lr_scheduler_fn is not None:
      self.agent.scheduler = optim.lr_scheduler.LambdaLR(self.agent.optim,self.scheduler.agent_lr_scheduler_fn)

  def _get_delta(
    self,
    agent1,
    agent2,
    batch,
    discount_rate,
    sample_weights,
    td_steps,
    optimise=True):

    #process batch
    S1 = torch.from_numpy(np.array([x[0] for x in batch])).float()
    A1 = torch.from_numpy(np.array([x[1] for x in batch])).long()
    R = torch.from_numpy(np.array([x[2] for x in batch])).float()
    S2 = torch.from_numpy(np.array([x[3] for x in batch])).float()
    T = torch.from_numpy(np.array([x[4] for x in batch])).float()

    with torch.no_grad():
      _, A2 = torch.max(agent1.forward(S2),1) #decide with q network

      Y = R.view(-1,1) + discount_rate**(td_steps)*agent2.forward(S2).gather(1,A2.view(-1,1))*T.view(-1,1) #evaluate with target network

    delta = Y - agent1.forward(S1).gather(1,A1.view(-1,1)) #optimise q network
    
    agent1.optim.zero_grad()
    if optimise:
      #optimise
      (sample_weights.view(-1,1)*delta**2).mean().backward()
      agent1.optim.step()

      if logging.getLogger().getEffectiveLevel() == logging.DEBUG: #additional logger calculations
        with torch.no_grad():
          delta2 = Y - agent1.forward(S1).gather(1,A1.view(-1,1)) #optimise q network
        logging.debug("mean loss improvement: {x}".format(x=torch.abs(delta).mean()-torch.abs(delta2).mean()))

    return torch.abs(delta).detach().numpy()

  def _step(self,agent,target,s1,eps,render=False):
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
    if render:
      self.env.render()
    sarst += (r,s2,1-int(done)) #t = termination signal (t=0 if s2 is terminal state, t=1 otherwise)

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
    td_steps = 1,
    reset=False,
    render = False):

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

    """
    logging.info("Running DQN.fit")
    if reset:
      self.scheduler.reset()
      self.agent.scheduler = optim.scheduler.LambdaLR(self.agent.opt,self.scheduler.agent_lr_scheduler_fn)
      self.trace = []

    scheduler = self.scheduler

    agent = self.agent
    target = copy.deepcopy(agent)
    # initialize and fill memory 
    memory = SumTree(max_memory_size)

    s1 = self.env.reset()
    memsize = 0
    step_list = deque(maxlen=td_steps)
    td = 0 #temporal difference step counter

    logging.info("Filling memory...")
    while memsize < max_memory_size:
      # fill step list
      step_list.append(self._step(agent,target,s1,scheduler.exploration_rate,False))
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


    # fit agent
    logging.info("Training...")
    for i in tqdm(range(n_episodes)):
          
      s1 = self.env.reset()

      ts_reward = 0

      td = 0
      step_list = deque(maxlen=td_steps)

      for j in range(max_ts_by_episode):

        #execute policy
        step_list.append(self._step(agent,target,s1,scheduler.exploration_rate,render))
        td += 1
        s1 = step_list[-1][3]
        r = step_list[-1][2] #latest reward

        if np.isnan(r):
          raise RuntimeError("The model diverged; decreasing step sizes or increasing tau can help to prevent this.")

        done = (step_list[-1][4] == 0)

        if td >= td_steps:
          # compute temporal difference n-steps SARST and its delta
          td_sarst = self._process_td_steps(step_list,discount_rate)
          delta = self._get_delta(agent,target,[td_sarst],discount_rate,torch.ones(1),td_steps,optimise=False)
          memory.add((delta[0] + 1.0/max_memory_size)**scheduler.PER_alpha,td_sarst)

        # sgd update
        # get replay batch
        P = memory.total()
        N = memory.get_current_size()
        if N > 0:
          for h in range(scheduler.steps_per_update):

            try:
              samples = np.random.uniform(high=P,size=min(scheduler.batch_size,N))
            except OverflowError as e:
              print(e)
              print("it seems that the model parameters are diverging. Decreasing step size or increasing tau might help.")

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
            delta = self._get_delta(agent,target,batch,discount_rate,batch_w,td_steps,optimise=True)

            #update memory
            for k in range(len(delta)):
              memory.update(batch_ids[k],(delta[k] + 1.0/max_memory_size)**scheduler.PER_alpha)

            #target network hard update
          if (scheduler.counter % scheduler.tau) == (scheduler.tau-1):
            target.model.load_state_dict(agent.model.state_dict())

        # trace information
        ts_reward += r

        # update learning rate and other hyperparameters
        agent._step()
        scheduler._step()

        if done:
          break

      if td < td_steps:
        td_sarst = self._process_td_steps(step_list,discount_rate)
        delta = self._get_delta(agent,target,[td_sarst],discount_rate,torch.ones(1),td_steps,optimise=False)
        memory.add((delta[0] + 1.0/max_memory_size)**scheduler.PER_alpha,td_sarst)

      self.mean_trace.append(ts_reward)
      logging.info("episode {n}, timestep {ts}, mean reward {x}".format(n=i,x=ts_reward,ts=scheduler.counter))

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
        action = np.argmax(self.agent.model.forward(obs).numpy())
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