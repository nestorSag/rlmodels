import random
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import itertools

class VanillaNet(nn.Module):
  """neural network with variale number and size of hidden layers. Uses ReLu for all of them and an allows specifying the activation function type of the output layer \n
  
  Parameters: \n
  layer_sizes (list of ints): list with hidden layer sizes \n
  input_size (int): input size \n
  output_size (int): output size \n
  final_activation (int): torch activation function for output layer. Can be None\n
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

#   def evaluate(self,x):
#     # this method is analog to forward but meant for evolutionary strategies, which often use less complicated output layers
#     # with non-standard transformations like rounding.
#     raise NotImplementedError("This method has to be implemented by its subclasses and it is meant for use in evolutionary strategies")

# class CartpoleNet(VanillaNet):
#   # network with an input size of 4 and output size of 1
#   # last activation is a sigmoid, every other one is relu
#   # layer sizes = list with layer sizes
#   def __init__(self,layer_sizes=[6],input_size=4,output_size=1,final_activation=torch.sigmoid):
#     super(CartpoleNet,self).__init__(layer_sizes,input_size,output_size,final_activation)

#   def evaluate(self,x):
#     if isinstance(x,np.ndarray):
#       x = torch.from_numpy(x).float()

#     y = self.forward(x).detach().numpy()[0]
#     return int(np.round(y)) #CMAES-formated output


# class PendulumNet(VanillaNet):
#   # network with an input size of 3 and output size of 1
#   # last activation is a tanh, every other one is relu
#   # layer sizes = list with layer sizes
#   def __init__(self,layer_sizes=[6],input_size=3,output_size=1,final_activation=torch.tanh):
#     super(PendulumNet,self).__init__(layer_sizes,input_size,output_size,final_activation)

#   def forward(self,x):
#     for layer in range(self.n_hidden_layers):
#       x = F.relu(getattr(self,"linear{n}".format(n=layer))(x))
#     x = 2*torch.tanh(getattr(self,"linear{n}".format(n=self.n_hidden_layers))(x))
#     return x

#   def evaluate(self,x):
#     if isinstance(x,np.ndarray):
#       x = torch.from_numpy(x).float()

#     y = self.forward(x).detach().numpy()[0]
#     #print("x: {x}, y: {y}".format(x=x,y=y))
#     return [y]

# class AcrobotNet(VanillaNet):
#   # network with default input size of 6 and output size of 1
#   # last activation is a sigmoid, every other one is relu
#   # layer sizes = list with layer sizes
#   def __init__(self,layer_sizes=[6],input_size=6,output_size=1,final_activation=torch.tanh):
#     super(AcrobotNet,self).__init__(layer_sizes,input_size,output_size,final_activation)

#   def evaluate(self,x):
#     if isinstance(x,np.ndarray):
#       x = torch.from_numpy(x).float()

#     y = self.forward(x).detach().numpy()[0]
#     #print("x: {x}, y: {y}".format(x=x,y=y))
#     return int(round(y))


# # class AirplaneNet(nn.Module):
# #   # network with an input size of 4 and output size of 1
# #   # last activation is a sigmoid, every other one is relu
# #   # layer sizes = list with layer sizes
# #   def __init__(self,layer_sizes=[6],n_rows=20,n_cols=6):
# #     super(AirplaneNet,self).__init__()
# #     self.n_hidden_layers = len(layer_sizes)
# #     layers = [2] + layer_sizes + [1]
# #     for layer in range(len(layers)-1):
# #         setattr(self,"linear{n}".format(n=layer),nn.Linear(layers[layer],layers[layer+1]))

# #     self.n_rows = n_rows
# #     self.n_cols = n_cols

# #   def forward(self,x):
# #     for layer in range(self.n_hidden_layers):
# #       x = F.relu(getattr(self,"linear{n}".format(n=layer))(x))
# #     x = F.relu(getattr(self,"linear{n}".format(n=self.n_hidden_layers))(x))
# #     return x

# #   def evaluate(self,obs):
# #     passengers = list(itertools.product(range(self.n_rows),range(self.n_cols)))
# #     scores = []
# #     for passenger in passengers:
# #       psng = np.array(passenger)
# #       psng = torch.from_numpy(psng).float()
# #       scores.append(self.forward(psng).detach().numpy()[0])

# #     sorted_passengers = [x for _,x in sorted(zip(scores,passengers), key = lambda x: x[0])]

# #     return sorted_passengers

# class MountainCarNet(nn.Module):
#   # network with default input size of 2 and output size of 1
#   # last activation is a sigmoid, every other one is relu
#   def __init__(self,layer_sizes=[6],input_size=2,output_size=1,final_activation=torch.sigmoid):
#     super(MountainCarNet,self).__init__(layer_sizes,input_size,output_size,final_activation)
  
#   def forward(self,x):
#     for layer in range(self.n_hidden_layers):
#       x = F.relu(getattr(self,"linear{n}".format(n=layer))(x))
#     x = 2*torch.sigmoid(getattr(self,"linear{n}".format(n=self.n_hidden_layers))(x))
#     return x

#   def evaluate(self,x):
#     if isinstance(x,np.ndarray):
#       x = torch.from_numpy(x).float()

#     y = self.forward(x).detach().numpy()[0]
#     #print("x: {x}, y: {y}".format(x=x,y=y))
#     return int(round(y))
