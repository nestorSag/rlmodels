import random
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import itertools

class VanillaNet(nn.Module):
  """neural network with variale number and size of hidden layers. Uses ReLu for all of them and an allows specifying the activation function type of the output layer 
  
  Parameters: 

  `layer_sizes` (`list` of `int`s): list with hidden layer sizes 

  `input_size` (`int`): input size 

  `output_size` (`int`): output size 

  `final_activation` (`int`): torch activation function for output layer. Can be None

  """


  def __init__(self,layer_sizes,input_size,output_size,final_activation):

    super(VanillaNet,self).__init__()
    self.n_hidden_layers = len(layer_sizes)
    layers = [input_size] + layer_sizes + [output_size]
    for layer in range(len(layers)-1):
        setattr(self,"linear{n}".format(n=layer),nn.Linear(layers[layer],layers[layer+1]))

    self.final_activation = final_activation

  def forward(self,x):
    if isinstance(x,np.ndarray):
      x = torch.from_numpy(x).float()

    for layer in range(self.n_hidden_layers):
      x = F.relu(getattr(self,"linear{n}".format(n=layer))(x))
    if self.final_activation is None:
      x = (getattr(self,"linear{n}".format(n=self.n_hidden_layers))(x))
    else:
      x = self.final_activation(getattr(self,"linear{n}".format(n=self.n_hidden_layers))(x))
    return x
