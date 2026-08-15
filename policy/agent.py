try:
  from .actor import Actor
  from .critic import Critic
except ImportError:
  from actor import Actor
  from critic import Critic
from torch import nn
from torch import optim
import random
import os
import torch
import math
import numpy as np
from torch.distributions import Normal
class Agent():
  def __init__(self, state_size, action_dim, env):
    self.actor = Actor(state_size, action_dim)
    self.critic = Critic(state_size)
    self.critic_loss = nn.SmoothL1Loss(beta=100)
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0001)
    self.buffer = []
    self.env = env
    self.state = env.reset()
    
    self.mean = [0] * 16
    self.sum_of_sd = [0] * 16
    self.observations = 0
  def welford(self, next_state):
    self.observations += 1
    for i in range(16):
      delta = float(next_state[i] - self.mean[i])
      self.mean[i] += delta / self.observations
      self.sum_of_sd[i] += delta * float((next_state[i] - self.mean[i]))
  def normalize(self, state):
    normalized = [0] * 16
    count = max(1, self.observations)
    for i in range(16):
        normalized[i] = (state[i] - self.mean[i]) / math.sqrt(self.sum_of_sd[i] / count + 1e-8)
    return normalized
  def train(self, iterations, gamma, LAMBDA, epsilon, c2):
    for iteration in range(iterations):
      self.collect_data()
      self.update_values(gamma, LAMBDA, epsilon, c2)
      
      rewards = [row["reward"] for row in self.buffer]
      dones = [row["done"] for row in self.buffer]
      
      episode_rewards = []
      current_ep_reward = 0
      for r, d in zip(rewards, dones):
          current_ep_reward += r
          if d:
              episode_rewards.append(current_ep_reward)
              current_ep_reward = 0

      avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
      if iteration % 50 == 0:
        self.save("agent_checkpoint.pt")
      print(f"Iteration {iteration} | Episodes: {len(episode_rewards)} | Avg episode reward: {avg_reward:.2f}")
    self.save("agent_checkpoint.pt")
  def collect_data(self):
    self.buffer = []
    for i in range(2048):
      self.welford(self.state)
      state_tensor = torch.tensor(self.normalize(self.state), dtype=torch.float32)
      with torch.no_grad():
        mean, log_std = self.actor(state_tensor)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(state_tensor).item()
      
      next_state, reward, done, terminated, _ = self.env.step(action.numpy())
      
      self.buffer.append({
        "state": state_tensor,
        "action": action,
        "log_prob": log_prob,
        "reward": reward,
        "value": value,
        "done": done,
        "terminated": terminated
      })
      
      self.state = next_state
      
      if done:
        self.state = self.env.reset()
        
  def gae_pass(self, gamma, LAMBDA):
    dones = [row["done"] for row in self.buffer]
    values = [row["value"] for row in self.buffer]
    rewards = [row["reward"] for row in self.buffer]
    terminate = [row["terminated"] for row in self.buffer]
    advantages = [0] * len(self.buffer)
    for i in reversed(range(len(self.buffer))):
      if i == len(self.buffer) - 1:
        with torch.no_grad():
          next_value = 0 if dones[i] else self.critic(torch.tensor(self.normalize(self.state), dtype=torch.float32)).item()
        next_advantage = 0
      else:
        next_value = values[i+1]
        next_advantage = advantages[i+1]

      td = rewards[i] + gamma * next_value * (1 - terminate[i]) - values[i]
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
        
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        critic_predictions = self.critic(batch_states).squeeze(-1)
        output = self.critic_loss(critic_predictions, batch_returns)
        self.critic_optimizer.zero_grad()
        output.backward()
        self.critic_optimizer.step()
  def save(self, path="agent_checkpoint.pt"):
    tmp = path + ".tmp"
    torch.save({
        "actor": self.actor.state_dict(),
        "critic": self.critic.state_dict(),
        "mean": self.mean,
        "sum_of_sd": self.sum_of_sd,
        "observations": self.observations
    }, tmp)
    os.replace(tmp, path)  # atomic rename: watchers see either the old or new file, never a partial one

  def load(self, path="agent_checkpoint.pt"):
    checkpoint = torch.load(path)
    self.actor.load_state_dict(checkpoint["actor"])
    self.critic.load_state_dict(checkpoint["critic"])
    self.mean = checkpoint["mean"]
    self.sum_of_sd = checkpoint["sum_of_sd"]
    self.observations = checkpoint["observations"]
