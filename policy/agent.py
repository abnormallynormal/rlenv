from policy.actor import Actor
from policy.critic import Critic
from torch import nn
from torch import optim
import random
import torch
import numpy as np
from torch.distributions import Normal
class Agent():
  def __init__(self, state_size, action_dim, env):
    self.actor = Actor(state_size, action_dim)
    self.critic = Critic(state_size)
    self.critic_loss = nn.SmoothL1Loss()
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0001)
    self.buffer = []
    self.env = env
    self.state = env.reset()
  def collect_data(self):
    self.buffer = []
    for i in range(2048):
      state_tensor = torch.tensor(self.state, dtype=torch.float32)
      with torch.no_grad():
        mean, log_std = self.actor(state_tensor)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(state_tensor).item()
      
      next_state, reward, done, _ = self.env.step(action.numpy())
      
      self.buffer.append({
        "state": state_tensor,
        "action": action,
        "log_prob": log_prob,
        "reward": reward,
        "value": value,
        "done": done
      })
      
      self.state = next_state
      
      if done:
        self.state = self.env.reset()
        
  def gae_pass(self, gamma, LAMBDA):
    dones = [row["done"] for row in self.buffer]
    values = [row["value"] for row in self.buffer]
    rewards = [row["reward"] for row in self.buffer]
    advantages = [0] * len(self.buffer)
    for i in reversed(range(len(self.buffer))):
      if i == len(self.buffer) - 1:
        with torch.no_grad():
          next_value = 0 if dones[i] else self.critic(torch.tensor(self.state, dtype=torch.float32)).item()
        next_advantage = 0
      else:
        next_value = values[i+1]
        next_advantage = advantages[i+1]
      td = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
      advantages[i] = td + gamma * LAMBDA * next_advantage * (1 - dones[i])
    
    advantages = np.array(advantages)
    returns = advantages + np.array(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns
  def update_values(self, gamma, LAMBDA, epsilon, c2):
    states = torch.stack([row["state"] for row in self.buffer])
    actions = torch.stack([row["action"] for row in self.buffer])
    old_log_probs = torch.stack([row["log_prob"] for row in self.buffer])
    advantages, returns = self.gae_pass(gamma, LAMBDA)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    for epoch in range(10):
      indices = [i for i in range(len(self.buffer))]
      random.shuffle(indices)
      batches = []
      for i in range(len(self.buffer) // 64):
        batches.append(indices[(i*64):(i*64 + 64)])
      
      for batch in batches:
        batch_states = states[batch]
        batch_actions = actions[batch]
        batch_old_log_probs = old_log_probs[batch]
        batch_advantages = advantages[batch]
        batch_returns = returns[batch]
        
        mean, log_std = self.actor(batch_states)
        std = log_std.exp()
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(batch_actions).sum(-1)
        
        ratios = torch.exp(new_log_probs - batch_old_log_probs)
        
        unclipped = ratios * batch_advantages
        clamped_ratios = ratios.clamp(1 - epsilon, 1 + epsilon)
        clipped = clamped_ratios * batch_advantages
        
        minimums = torch.minimum(unclipped, clipped)
        policy_loss = -torch.mean(minimums)
        
        entropy = torch.mean(torch.sum(dist.entropy(), dim=-1))
        
        actor_loss = policy_loss - c2 * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        critic_predictions = self.critic(batch_states).squeeze(-1)
        output = self.critic_loss(critic_predictions, batch_returns)
        self.critic_optimizer.zero_grad()
        output.backward()
        self.critic_optimizer.step()
