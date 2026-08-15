# RL Arcade

RL Arcade is a standalone Windows demo of the reinforcement-learning agents in this repository. Reviewers can run the compiled executables without installing Python or any dependencies. **Comparison** mode runs a trained DQN beside a random baseline. **Live training** runs the original learner in a background thread while recorded episodes play in the interface.

## Release files

- `RLArcade.exe` contains Flappy Bird and Snake, the original Dueling Double DQN models, trained PyTorch checkpoints, visual playback, and metrics.
- `BipedDemo.exe` contains the original MuJoCo biped, PPO actor checkpoint, observation normalization statistics, robot XML, and native live viewer.

Place both files in the same directory. **Watch MuJoCo** opens the saved PPO policy. MuJoCo training is intentionally not exposed in the arcade because it is CPU-intensive.

## Bundled models

- Flappy Bird: 5,000 episodes, final epsilon 0.01, evaluation average 13.90 over 100 fixed-seed episodes.
- Snake: intentionally stopped at episode 1,100 after epsilon reached 0.01, evaluation average 28.26 over 100 fixed-seed episodes.
- Random-policy evaluation average: 0.05 in both environments.

Both use the repository's original Dueling Double DQN, Double DQN target calculation, prioritized replay (`alpha=0.6`, `beta=0.4`, increment `0.001`), Adam (`lr=0.0001`), Huber loss, `gamma=0.99`, batch size 32, and target synchronization every 5,000 steps.

Live DQN training is always user-started, begins from scratch, and stops automatically at 5,000 episodes. It uses initial epsilon `0.9999`, minimum epsilon `0.01`, Flappy decay `0.999`, and Snake decay `0.995`. It never overwrites the bundled evaluation models; new checkpoints are stored under `%LOCALAPPDATA%\RLArcade`.

The finalized Windows executables are stored under `dist/`. The temporary PyInstaller build configuration and environments were removed after both artifacts passed launch tests.
