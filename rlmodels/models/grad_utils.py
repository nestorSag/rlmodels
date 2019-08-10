import torch
import torch.optim as optim
import numpy as np

class SumTree:
  """efficient memory data sctructure class (fast retrieves and updates).

  source of the SumTree class code : https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
  
  **Parameters**:

  *capacity* (*int*): number of tree leaves

  """
  write = 0
  current_size=0

  def __init__(self, capacity):
    # INPUT
    # *capacity*: number of tree leaves
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

  
  def total(self):
    """returns the sum of leaf weights
    """
    return self.tree[0]

  def add(self, p, data):
    """adds data to tree, potentially overwritting older data
  
    **Parameters**:

    *p* (*float*): leaf weight

    *data*: leaf data
    """
  
    idx = self.write + self.capacity - 1

    self.data[self.write] = data
    self.update(idx, p)

    self.write += 1
    if self.write >= self.capacity:
      self.write = 0

    self.current_size = min(self.current_size+1,self.capacity)

  def update(self, idx, p):
    """updates leaf weight
    
    **Parameters**:

    *idx* (*int*): leaf index
    
    *p* (*float*): new weight

    """
    change = p - self.tree[idx]

    self.tree[idx] = p
    
    if self.capacity > 1:
      self._propagate(idx, change)

  def get(self, s):
    """get leaf corresponding to numeric value
    
    **Parameters**:

    *s* (*float*): numeric value

    **Returns**:

    triplet with leaf id (*int*), tree node id (*int*) and  leaf data
    """

    idx = self._retrieve(0, s)
    dataIdx = idx - self.capacity + 1

    return (idx, self.tree[idx], self.data[dataIdx])

class Agent(object):
  """neural network gradient optimisation wrapper
  
  **Parameters**:

  *model* (*torch.nn.Module*): Pytorch neural network model

  *opt* (*torch.optim*): Pytorch optimizer object 

  """
  
  def __init__(self,model,opt=None):
    self.model = model
    self.optim = opt
    self.scheduler = None

  def forward(self,x):
    """ Apply *model*'s  *forward* method to input

    **Parameters**:

    *x* (*torch.Tensor*): state tensor

    """

    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()
    return self.model.forward(x)

  def _step(self):
    if self.scheduler is not None:
      self.scheduler.step()

# class FormattedActionEnv(object):
#   """environment wrapper that formats a model output action to gym environment's required format
  
#   Parameters:

#   *env* : environment with the same interface as in gym library

#   *action_map* (*function*): mapper that takes a model output and formats it to the environment's standard input type

#   """
#   def __init__(self,env,action_map):
#     self.env = env
#     self.action_space = self.env.action_space
#     self.observation_space = self.env.observation_space
#     self.action_map = action_map

#   def step(self,a):
#     a = self.action_map(a)
#     s,r,done,info = self.env.step(a)
#     return s,r,done,info

#   def render(self):

#     self.env.render()
#   def close(self):
#     self.env.close()

#   def reset(self):
#     self.env.reset()

#   def seed(self,seed):
#     self.env.seed(seed)