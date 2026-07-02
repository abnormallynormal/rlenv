import torch
import torch.nn as nn

class Critic(nn.Module):
  def __init__(self, state_size):
    super(Critic, self).__init__()
    self.hidden_1 = nn.Linear(state_size, 64)
    self.hidden_2 = nn.Linear(64, 64)
    self.value = nn.Linear(64, 1)
    
    self.init_weights()    
  def init_weights(self):
    for layer in [self.hidden_1, self.hidden_2]:
      nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
      nn.init.constant_(layer.bias, 0.0)
    nn.init.orthogonal_(self.value.weight)
    nn.init.constant_(self.value.bias, 0.0)
  def forward(self, x):
    x = torch.tanh(self.hidden_1(x))
    x = torch.tanh(self.hidden_2(x))
      
    evaluation = self.value(x)
    return evaluation