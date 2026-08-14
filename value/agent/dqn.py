import random
import torch
from torch import nn
from torch import optim
try:
  from .network import Network
except ImportError:  # Preserve direct execution from the value directory.
  from agent.network import Network

class Agent:
  def __init__(self, state_size, action_size):
    self.training_network = Network(state_size, action_size)
    self.frozen_network = Network(state_size, action_size)
    self.loss = nn.SmoothL1Loss(reduction="none")
    self.optimizer = optim.Adam(self.training_network.parameters(), lr=0.0001)
    self.frozen_network.load_state_dict(self.training_network.state_dict())
    self.steps = 0
  
  def select_action(self, epsilon, state, action_size):
    self.steps += 1
    r = random.random()
    if(r < epsilon):
      action = random.randint(0, action_size - 1)
      return action
    else:
      state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
      output = self.training_network(state_tensor)
      return torch.argmax(output).item()
    
  
  def sync_networks(self):
    self.frozen_network.load_state_dict(self.training_network.state_dict())
    
  def train(self, batch, is_weights, gamma):
    states, actions, rewards, state_primes, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    state_primes = torch.tensor(state_primes, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    is_weights = torch.tensor(is_weights, dtype=torch.float32)
        
    with torch.no_grad():
      best_actions = self.training_network(state_primes).argmax(dim=1, keepdim=True)
      q_values = self.frozen_network(state_primes)
      q_primes = q_values.gather(1, best_actions).squeeze(1)
      
    target = rewards + gamma * q_primes * (1 - dones)
    predictions = self.training_network(states)    
    actions = actions.unsqueeze(1)
    predictions = predictions.gather(1, actions).squeeze(1)
    output = (is_weights * self.loss(predictions, target)).mean()    
    td_errors = torch.abs(target - predictions).detach()
    
    self.optimizer.zero_grad()
    output.backward()
    self.optimizer.step()
        
    return td_errors

