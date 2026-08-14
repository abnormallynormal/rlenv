# RL Arcade

RL Arcade is a standalone Windows demo of the reinforcement-learning agents in this repository. Reviewers can run the compiled executables without installing Python or any dependencies.

## Release files

- `RLArcade.exe` contains Flappy Bird and Snake, the original Dueling Double DQN implementation, prioritized replay, trained PyTorch checkpoints, visual playback, metrics, and in-app training.
- `BipedDemo.exe` contains the original MuJoCo biped, PPO actor checkpoint, observation normalization statistics, robot XML, and native live viewer.

Place both files in the same directory. The arcade's **Launch MuJoCo Biped** button opens the companion automatically.

## Bundled models

- Flappy Bird: 5,000 episodes, final epsilon 0.01, evaluation average 13.90 over 100 fixed-seed episodes.
- Snake: intentionally stopped at episode 1,100 after epsilon reached 0.01, evaluation average 28.26 over 100 fixed-seed episodes.
- Random-policy evaluation average: 0.05 in both environments.

Both use the repository's original Dueling Double DQN, Double DQN target calculation, prioritized replay (`alpha=0.6`, `beta=0.4`, increment `0.001`), Adam (`lr=0.0001`), Huber loss, `gamma=0.99`, batch size 32, and target synchronization every 5,000 steps.

## Training

Run independent, resumable jobs with checkpoints every 100 episodes:

```powershell
py -3.11 train_one.py flappy
py -3.11 train_one.py snake
```

Flappy uses `epsilon=max(0.01, 0.9999 * 0.999^episode)`. Snake uses `epsilon=max(0.01, 0.9999 * 0.995^episode)`.

## Build

On Windows with Python 3.11 and the project dependencies installed:

```powershell
.\build.ps1
.\build-biped.ps1
```

The resulting files are written to `dist/`. PyInstaller must run separately on each target operating system; these scripts build Windows executables.
