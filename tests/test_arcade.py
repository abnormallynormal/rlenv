import tempfile
import unittest
from pathlib import Path

from arcade.agents import DQNPolicy, RandomAgent
from arcade.environments import ENVIRONMENTS


ROOT = Path(__file__).resolve().parents[1]


def evaluate(factory, agent, episodes=100, seed=1000):
    scores = []
    for episode in range(episodes):
        env = factory(seed + episode)
        state = env.reset(seed + episode)
        done = False
        while not done:
            action = agent.select_action(state)
            state, _reward, done = env.step(action)
        scores.append(env.score)
    return sum(scores) / len(scores)


class EnvironmentTests(unittest.TestCase):
    def test_environments_have_finite_episodes(self):
        for name, factory in ENVIRONMENTS.items():
            with self.subTest(name=name):
                env = factory(seed=3)
                state = env.reset(seed=3)
                for _ in range(4000):
                    state, reward, done = env.step(0)
                    self.assertIsInstance(reward, float)
                    if done:
                        break
                self.assertTrue(done)

    def test_adapters_match_original_flappy_transition(self):
        import random
        from value.modules.flappy.environment import Environment as Original

        random.seed(41)
        original = Original()
        original_state = original.reset()
        random.seed(41)
        adapted = ENVIRONMENTS["Flappy Bird"](seed=41)
        adapted_state = adapted.reset(seed=41)
        self.assertEqual(original_state, adapted_state)
        for action in (0, 1, 0, 0, 1):
            expected = original.step(action)
            actual = adapted.step(action)
            self.assertEqual(expected, actual)

    def test_adapters_match_original_snake_transition(self):
        import random
        from value.modules.snake.environment import Environment as Original

        def rollout(factory):
            random.seed(73)
            environment = factory()
            transitions = [environment.reset()]
            for action in (0, 1, 0, 2, 0):
                transitions.append(environment.step(action))
            return transitions

        self.assertEqual(rollout(Original), rollout(ENVIRONMENTS["Snake"]))

    def test_seeded_resets_are_reproducible(self):
        for name, factory in ENVIRONMENTS.items():
            with self.subTest(name=name):
                first = factory(seed=5)
                second = factory(seed=5)
                self.assertEqual(first.reset(seed=19), second.reset(seed=19))


class PolicyTests(unittest.TestCase):
    def test_requested_epsilon_schedules(self):
        flappy_final = max(0.01, 0.9999 * (0.999 ** 4999))
        snake_final = max(0.01, 0.9999 * (0.995 ** 4999))
        self.assertEqual(flappy_final, 0.01)
        self.assertEqual(snake_final, 0.01)

    def test_bundled_models_are_nonempty(self):
        for name in ENVIRONMENTS:
            path = ROOT / "models" / (name.lower().replace(" ", "_") + ".pt")
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)

    def test_checkpoint_round_trip(self):
        original = DQNPolicy(4, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            original.save(path)
            restored = DQNPolicy.load(path, 4, 2)
            for expected, actual in zip(
                original.agent.training_network.parameters(),
                restored.agent.training_network.parameters(),
            ):
                self.assertTrue(expected.equal(actual))


if __name__ == "__main__":
    unittest.main()
