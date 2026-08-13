from collections import deque
from agent.sumtree import SumTree
import random

class Buffer:
  def __init__(self, alpha, beta, beta_increment, epsilon):
    self.memory = SumTree(50000)
    self.alpha = alpha
    self.beta = beta
    self.beta_increment = beta_increment
    self.epsilon = epsilon
    self.max_priority = 1
    print(self.memory.size)
  def add_entry(self, s_t, a_t, r_t, s_t1, done):
    priority = self.max_priority ** self.alpha
    self.memory.add(priority, (s_t, a_t, r_t, s_t1, done))
  def select_random(self, B):
    segment = self.memory.get_total() / B
    
    samples = []
    priorities = []
    indices = []
    
    for i in range(B):
      low = segment * i
      high = segment * (i + 1)
      value = random.uniform(low, high)
      data_index, priority, transition = self.memory.get_leaf(value)
      priorities.append(priority)
      samples.append(transition)
      indices.append(data_index)
    
    total = self.memory.get_total()
    probabilities = [p / total for p in priorities]
    weights = [(1 / (len(self.memory) * prob)) ** self.beta for prob in probabilities]
    max_weight = max(weights)
    weights = [w / max_weight for w in weights]
    return indices, samples, weights
  
  def update_priorities(self, indices, td_errors):
    for index, td_error in zip(indices, td_errors):
      priority = td_error + self.epsilon
      leaf_index = index + self.memory.size - 1
      self.memory.update(leaf_index, priority ** self.alpha)
      self.max_priority = max(self.max_priority, priority)
    
  def step_beta(self):
    self.beta = min(1.0, self.beta + self.beta_increment)
  
  def __len__(self):
    return len(self.memory)