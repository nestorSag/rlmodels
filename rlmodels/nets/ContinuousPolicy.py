import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as normal
from .FullyConnected import FullyConnected

class ContinuousPolicy(FullyConnected):
  """ continuous policy network where each action component is normally distributed with no correlation between components.
  Hidden layers have ReLU activation by default. The `forward` method returns a `torch.distributions.normal.Normal` instance.
  
  **Parameters**: 

  *layer_sizes* (*list* of *int*s): list with hidden layer sizes 

  *input_size* (*int*): state dimension

  *output_size* (*int*): action dimension 

  *activation_fn* (*function*): activation function for the mean

  *min_sd* (*float*): minimum standard deviation value

  """
  def __init__(self,layer_sizes,input_size,output_size,mean_activation_fn,min_sd=1e-1):

    super(ContinuousPolicy,self).__init__(layer_sizes,input_size,2*output_size,None)
    self.d = output_size
    self.mean_activation_fn = mean_activation_fn
    self.min_sd=min_sd

  def forward(self,x):
    x = super(ContinuousPolicy,self).forward(x)

    if len(x.shape) > 1:
      mean = self.mean_activation_fn(x[:,0:self.d])
      sd = torch.exp(x[:,self.d:(2*self.d)])
    else:
      mean = self.mean_activation_fn(x[0:self.d])
      sd = torch.exp(x[self.d:(2*self.d)])

    return normal.Normal(mean,torch.clamp(sd,min=self.min_sd,max=np.Inf))