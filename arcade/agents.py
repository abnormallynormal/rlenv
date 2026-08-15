"""Thin product adapters around the project's original DQN implementation."""

import random
from pathlib import Path

import torch

from value.agent.dqn import Agent as OriginalDQN


class RandomAgent:
    label = "Random agent"

    def __init__(self, action_count, seed=None):
        self.action_count = action_count
        self.random = random.Random(seed)

    def select_action(self, state):
        return self.random.randrange(self.action_count)


class DQNPolicy:
    """Inference/checkpoint facade; learning still happens in OriginalDQN."""

    label = "Trained DQN"

    def __init__(self, state_size, action_size, agent=None):
        self.state_size = state_size
        self.action_size = action_size
        self.agent = agent or OriginalDQN(state_size, action_size)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.agent.training_network(state_tensor).argmax(dim=1).item()

    @classmethod
    def load(cls, path, state_size, action_size):
        policy = cls(state_size, action_size)
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        policy.agent.training_network.load_state_dict(checkpoint["training_network"])
        policy.agent.frozen_network.load_state_dict(
            checkpoint.get("frozen_network", checkpoint["training_network"])
        )
        policy.agent.steps = int(checkpoint.get("steps", 0))
        policy.agent.training_network.eval()
        policy.agent.frozen_network.eval()
        return policy
