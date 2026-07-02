import torch
import torch.nn as nn

class Actor(nn.Module):
  def __init__(self, state_size, action_dim):
    super(Actor, self).__init__()
    self.hidden_1 = nn.Linear(state_size, 64)
    self.hidden_2 = nn.Linear(64, 64)
    self.distributions = nn.Linear(64, action_dim*2)
    
    self.init_weights()
  def init_weights(self):
    for layer in [self.hidden_1, self.hidden_2]:
      nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
      nn.init.constant_(layer.bias, 0.0)
    nn.init.orthogonal_(self.distributions.weight, gain=0.01)
    nn.init.constant_(self.distributions.bias, 0.0)
    
  def forward(self, x):
    x = torch.tanh(self.hidden_1(x))
    x = torch.tanh(self.hidden_2(x))
    probability_distributions = self.distributions(x)
    means, log_stds = probability_distributions.chunk(2, dim=-1)
    log_stds = torch.clamp(log_stds, -20, 2)    
    return means, log_stds