"""Demo adapters around the project's original game environments.

The adapters add deterministic seeding, display metadata, and state-key
discretization. Physics, rewards, observations, collision handling, frame
skipping, and action meanings remain in the original environment modules.
"""

import random

from value.modules.flappy.environment import Environment as OriginalFlappy
from value.modules.snake.environment import Environment as OriginalSnake


class FlappyEnvironment(OriginalFlappy):
    name = "Flappy Bird"
    width = OriginalFlappy.grid_width
    height = OriginalFlappy.grid_height
    action_names = ("glide", "flap")

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        super().__init__()
        self.steps = 0
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.steps = 0
        return super().reset()

    def step(self, action, frame_skip=6):
        state, reward, done = super().step(action, frame_skip=frame_skip)
        self.steps += 1
        return state, float(reward), done

    @staticmethod
    def state_key(state):
        y, velocity, distance, gap = state
        return (
            min(11, max(0, int(y * 12))),
            min(11, max(0, int(velocity * 12))),
            min(7, max(0, int(distance * 8))),
            min(11, max(0, int(gap * 12))),
        )


class SnakeEnvironment(OriginalSnake):
    name = "Snake"
    action_names = ("straight", "left", "right")

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        super().__init__()
        self.steps = 0

    @property
    def body(self):
        """Display alias; the original environment calls this snake_coordinates."""
        return self.snake_coordinates

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        state = super().reset()
        self.steps = 0
        return state

    def step(self, action):
        state, reward, done = super().step(action)
        self.steps += 1
        return state, float(reward), done

    @staticmethod
    def state_key(state):
        return sum((1 << index) for index, enabled in enumerate(state) if enabled)


ENVIRONMENTS = {"Flappy Bird": FlappyEnvironment, "Snake": SnakeEnvironment}
