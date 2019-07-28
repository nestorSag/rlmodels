# rlmodels: a reinforcement learning library

This project is a collection of some popular optimisation algorithms for reinforcement learning problem. At the moment the available models are:

* Double Q network with prioritazed experience replay (PER)
* Covariance matrix adaptive evolutionary strategy (CMAES)

with some more going to be added in the future.

It works with Pytorch models and environment classes like the OpenAI gym ones. Any environment class wrapper that mimic their basic functionality should be fine, but more on that below.

## Getting Started

### Prerequisites

The project uses ```python 3.6``` and ```torch 1.1.0```.

### Installing

It can be installed directly from pip like 
```bash
pip install rlmodels
```

## Usage

Below is a summary of how the program works. **To see the full documentation click [here](https://nestorsag.github.io/rlmodels/index.html#package)**

### Initialization

The following is an example with the popular CartPole environment using a double Q network. First the setup

```python
import numpy as np
import torch
import torch.optim as optim
import gym

from rlmodels.models.grad import *
from rlmodels.nets import VanillaNet
```

The models are divided in evolutionary strategies (es) and gradient-based ones (grad). The library also has a basic network definition, VanillaNet, to which we only need to specify number and size of hidden layer, input and output sizes, and last activation function. It uses ReLU everywhere else by default.

let's create the basic objects 

```python
# change max episodic ts to 300
max_ep_ts = 300

env = gym.make('CartPole-v0')
env._max_episode_steps = max_ep_ts

env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

agent = VanillaNet([60],4,2,None)
target = VanillaNet([60],4,2,None)

# set hyperparameter runtime schedule as a function of the global number of timesteps
ddq_scheduler = DoubleQNetworkScheduler(
	batch_size = lambda t: 200, #constant
	exploration_rate = lambda t: max(0.01,0.05 - 0.01*int(t/2500)), #decrease exploration down to 1% after 10,000 steps
	PER_alpha = lambda t: 1, #constant
	PER_beta = lambda t: 1, #constant
	tau = lambda t: 100, #constant
	learning_rate_update = lambda t: 1.25**(-int(t/2500)), #multiplicative update. decrease step size every 2,500 steps,
	sgd_update = lambda t: 1) #number of steps between sgd updates

ddq = DoubleQNetwork(agent,target,env,ddq_scheduler)

```

the models take a scheduler object as argument which allows parameters to be changed at runtime accordint to user-defined rules. For example reducing learning rate and exploration rate after a certain number of iterations, as above. Now we can train the model

```python
ddq.fit(
	n_episodes=500,
	max_ts_by_episode=max_ep_ts,
	initial_learning_rate=1,
	max_memory_size=2000,
	verbose=True)
```

Once the agent is trained we can visualize the reward trace. If we are using an environment with a render method (like OpenAI ones) we can also visualise the trained agent. We can also use the trained model using the ```forward``` method of the ```ddq``` object or simply extract it with ```ddq.agent```

```python
ddq.plot() #plot reward traces
ddq.play(n=200) #observe the agent play
```

see the ```example``` folder for an analogous use of CMAES.

### Environment
For custom environments or custom rewards, its possible to make a wrapper tha mimics te behavior of the step() and reset() function of gym's environemnts
```python
class MyCustomEnv(object):
	def __init__(self,env):
		self.env = env
	def step(self,action):
		## get next state s, reward, termination flag (boolean) and any additional info
		return s,r, terminated, info #need to output these 4 things (info can be None)
	def reset(self):
		#something
	def seed(self):
		#something
```
