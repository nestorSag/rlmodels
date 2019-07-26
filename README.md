# rlmodels: Out-of-the-box reinforcement learning

This project is a collection of some popular reinforcement learning algorithms. At the moment the available models are:

* Double Q network with prioritazed experience replay (PER)
* Covariance matrix adaptive evolutionary strategy (CMAES)

with some more going to be added in the future.

It works with Pytorch models and environment classes like the OpenAI gym ones. Any environment class wrapper that mimic their basic functionality should be fine, but more on that below.

## Getting Started

### Prerequisites

The projectu uses ```python 3.6``` and ```torch 1.1.0```. If not installed, it downloads the CPU version.

### Installing

It can be installed directly from pip like 
```bash
pip install rlmodels
```

## Usage

Below is a summary of how the program works. To see the full documentation click [here](https://nestorsag.github.io/streaming-gmm/index.html#package)

### Initialization

Below is an example with the popular CartPole environment using a double Q network. First the setup

```python

from rlmodels.models.grad import DoubleQNetwork
from rlmodels.nets import VanillaNet
import gym
```

the models are divided in evolutionary strategies (es) and gradient-based ones (grad). The library also has a basic network deffinition, VanillaNet, to which we only need to specify number and size of hidden layer, input and output sizes, and last activation function. It uses ReLu everywhere else by default.

let's create the basic objects 

```python
env = gym.make('CartPole-v0')

agent = VanillaNet([60],4,2,None)
target = VanillaNet([60],4,2,None)

##make it reproducible
env.seed(1)
np.random.seed(1)
torch.manual_seed(1)

ddq = DoubleQNetwork(agent,target,env)
```

now we can fit the agent

```python
ddq.fit(n_episodes=1000,
	max_ts_by_episode=200,
	batch_size=lambda t: 200,
	exploration_rate_func = lambda t: max(0.01,0.05 - 0.01*int(t/2500)),
	max_memory_size=2000,
	learning_rate=0.001,
	tau=lambda t: 100,
	scheduler_func=lambda t: 1.25**(-int(t/2500)),
	verbose=True)
```

almost all arguments receive a function that maps timesteps to parameter values, to allow for dynamic tunning, for example to decrease stepsize and exploration rate after a fixed number of steps, as above.

Once the agent is trained we can visualize the reward trace. If we are using an environment with a render method (like OpenAI ones) we can also visualise the trained agent.

```python
ddq.plot()
ddq.play(n=200)
```

### Environment
for custom environments or custom rewards, its possible to make a wrapper tha mimics te behavior of the step() and reset() function of gym's environemnts
```python
class MyCustomEnv(object):
	def __init__(self,env):
		self.env = env
	def step(self,action):
		## get next state s, reward, and termination flag if needed
		return s,r, termination, None #need to output 4 things
	def reset(self):
		pass
	def seed(self):
		pass
```

## Authors

* **Nestor Sanchez - nestor.sag@gmail.com**