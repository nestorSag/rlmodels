import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FullyConnected import FullyConnected

class DiscretePolicy(FullyConnected):
  """ discrete policy network where actions are taken according to a softmax function. 
  Hidden layers have ReLU activation by default. The `forward` method returns a `torch.distributions.Categorical` instance.
  
  **Parameters**: 

  *layer_sizes* (*list* of *int*s): list with hidden layer sizes 

  *input_size* (*int*): state dimension

  *output_size* (*int*): action dimension

  """
  def __init__(self,layer_sizes,input_size,output_size):

    super(DiscretePolicy,self).__init__(layer_sizes,input_size,output_size,F.softmax)

  def forward(self,x):
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()

    for layer in range(self.n_hidden_layers):
      x = F.relu(getattr(self,"linear{n}".format(n=layer))(x))
    x = self.final_activation(getattr(self,"linear{n}".format(n=self.n_hidden_layers))(x),dim=-1)
    
    dist = torch.distributions.Categorical(x)
    return dist