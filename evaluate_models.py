from pathlib import Path

import torch

from arcade.agents import DQNPolicy, RandomAgent
from arcade.environments import ENVIRONMENTS


def evaluate(factory, policy, episodes=100, max_steps=5000):
    scores = []
    for episode in range(episodes):
        environment = factory(seed=1000 + episode)
        state = environment.reset(seed=1000 + episode)
        for _ in range(max_steps):
            action = policy.select_action(state)
            state, _reward, done = environment.step(action)
            if done:
                break
        scores.append(environment.score)
    return sum(scores) / len(scores), max(scores)


def main():
    torch.set_num_threads(1)
    root = Path(__file__).parent / "models"
    for name, factory in ENVIRONMENTS.items():
        environment = factory(seed=1)
        state_size = len(environment.reset(seed=1))
        action_size = len(environment.action_names)
        filename = name.lower().replace(" ", "_") + ".pt"
        trained = DQNPolicy.load(root / filename, state_size, action_size)
        random = RandomAgent(action_size, seed=3)
        trained_average, trained_max = evaluate(factory, trained)
        random_average, random_max = evaluate(factory, random)
        print(
            f"{name}: trained avg={trained_average:.2f}, max={trained_max}; "
            f"random avg={random_average:.2f}, max={random_max}"
        )


if __name__ == "__main__":
    main()
