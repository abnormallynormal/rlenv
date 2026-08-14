"""Thin product adapters around the project's original DQN implementation."""

import random
from pathlib import Path

import torch

from value.agent.buffer import Buffer
from value.agent.dqn import Agent as OriginalDQN
from value.train import train_loop


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

    def save(self, path, metadata=None):
        torch.save(
            {
                "training_network": self.agent.training_network.state_dict(),
                "frozen_network": self.agent.frozen_network.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
                "steps": self.agent.steps,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "metadata": metadata or {},
            },
            path,
        )

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


def train_agent(
    environment_factory,
    episodes,
    seed=7,
    progress=None,
    epsilon_start=0.9999,
    epsilon_min=0.01,
    epsilon_decay=0.999,
    checkpoint_path=None,
    checkpoint_every=100,
):
    """Train using the same Dueling Double DQN + PER loop as value/main.py."""
    random.seed(seed)
    torch.manual_seed(seed)
    environment = environment_factory(seed)
    state_size = len(environment.reset(seed))
    action_size = len(environment.action_names)
    dqn = OriginalDQN(state_size, action_size)
    replay_buffer = Buffer(
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        epsilon=1e-5,
    )
    epsilon = epsilon_start
    target_sync_counter = 0
    scores = []
    start_episode = 0

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        dqn.training_network.load_state_dict(saved["training_network"])
        dqn.frozen_network.load_state_dict(saved["frozen_network"])
        dqn.optimizer.load_state_dict(saved["optimizer"])
        dqn.steps = saved["agent_steps"]
        replay_buffer = saved["replay_buffer"]
        epsilon = saved["epsilon"]
        target_sync_counter = saved["target_sync_counter"]
        scores = saved["scores"]
        start_episode = saved["episode"]
        random.setstate(saved["python_random_state"])
        torch.set_rng_state(saved["torch_random_state"])

    for episode in range(start_episode, episodes):
        target_sync_counter, _unused_epsilon, _reward, score = train_loop(
            environment,
            dqn,
            replay_buffer,
            epsilon,
            action_size,
            target_sync_counter,
            recorder=None,
        )
        scores.append(score)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        completed = episode + 1
        if checkpoint_path and (completed % checkpoint_every == 0 or completed == episodes):
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(
                {
                    "episode": completed,
                    "training_network": dqn.training_network.state_dict(),
                    "frozen_network": dqn.frozen_network.state_dict(),
                    "optimizer": dqn.optimizer.state_dict(),
                    "agent_steps": dqn.steps,
                    "replay_buffer": replay_buffer,
                    "epsilon": epsilon,
                    "target_sync_counter": target_sync_counter,
                    "scores": scores,
                    "python_random_state": random.getstate(),
                    "torch_random_state": torch.get_rng_state(),
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        if progress and (episode + 1) % max(1, episodes // 20) == 0:
            recent = scores[-100:]
            progress(episode + 1, episodes, sum(recent) / len(recent), epsilon)

    dqn.training_network.eval()
    dqn.frozen_network.eval()
    return DQNPolicy(state_size, action_size, dqn), scores, epsilon
