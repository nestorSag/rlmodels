import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal

class ContinuousPolicy(nn.Module):
  """ continuous policy network where each action component is normally distributed with no correlation between components. Hidden layers have ReLU activation by default.
  
  Parameters: 

  `layer_sizes` (`list` of `int`s): list with hidden layer sizes 

  `input_size` (`int`): state dimension

  `output_size` (`int`): action dimension 

  """


  def __init__(self,layer_sizes,input_size,output_size):

    super(ContinuousPolicy,self).__init__()
    self.n_hidden_layers = len(layer_sizes)
    layers = [input_size] + layer_sizes + [2*output_size] #each output has 2 parameters: mean and standard deviation
    for layer in range(len(layers)-1):
        setattr(self,"linear{n}".format(n=layer),nn.Linear(layers[layer],layers[layer+1]))

    self.mu_layer = nn.Linear(layers[len(layers)-2],output_size)
    self.sd_layer = nn.Linear(layers[len(layers)-2],output_size)
    self.d = output_size

  def forward(self,x):
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()

    for layer in range(self.n_hidden_layers):
      x = F.relu(getattr(self,"linear{n}".format(n=layer))(x))

    mean = self.mu_layer(x)
    sd = torch.exp(self.sd_layer(x))

    return torch.cat((mean,sd),dim=-1)

  def get_action_dist(self,p):
    """ Get action distribution given a state 
    
    Parameters:

    `s` (`torch.Tensor`) state tensor

    Returns (`torch.distributions.categorical.Categorical`) categorical distribution

    """
    p = p.view(-1,2*self.d)
    a_mean = p[:,0:self.d]
    a_sd = p[:,self.d:(2*self.d)]

    pi = normal.Normal(a_mean,a_sd)
    return pi