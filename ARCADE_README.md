# RL Arcade

RL Arcade is a standalone Windows demo of the reinforcement-learning agents in this repository. Reviewers can run the compiled executables without installing Python or any dependencies.

## Release files

- `RLArcade.exe` contains Flappy Bird and Snake, the original Dueling Double DQN models, trained PyTorch checkpoints, visual playback, and metrics.
- `BipedDemo.exe` contains the original MuJoCo biped, PPO actor checkpoint, observation normalization statistics, robot XML, and native live viewer.

Place both files in the same directory. The arcade's **Launch MuJoCo Biped** button opens the companion automatically.

## Bundled models

- Flappy Bird: 5,000 episodes, final epsilon 0.01, evaluation average 13.90 over 100 fixed-seed episodes.
- Snake: intentionally stopped at episode 1,100 after epsilon reached 0.01, evaluation average 28.26 over 100 fixed-seed episodes.
- Random-policy evaluation average: 0.05 in both environments.

Both use the repository's original Dueling Double DQN, Double DQN target calculation, prioritized replay (`alpha=0.6`, `beta=0.4`, increment `0.001`), Adam (`lr=0.0001`), Huber loss, `gamma=0.99`, batch size 32, and target synchronization every 5,000 steps.

The finalized Windows executables are stored under `dist/`. The temporary PyInstaller build configuration and environments were removed after both artifacts passed launch tests.
