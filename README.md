# RL Arcade

> A collection of reinforcement-learning agents learning to complete tasks across custom game environments and a continuous-control robotics simulation.

![RL agents in action, playing Flappy Bird, Snake, and controlling a MuJoCo biped.](rlenv.png)
## What is this?

This project is a collection of reinforcement-learning agents, each placed in a unique environment and trained to perform a different task:

- An environment where an agent learns to play Snake
- An environment where an agent learns to play Flappy Bird
- An environment where an agent learns to control various motors on a custom MuJoCo biped robot to walk

Both Snake and Flappy Bird were custom environments I built, including custom physics, collision detection, state observations, reward functions. I used PyGame to render graphics for the user to see how the agent was learning. 

Throughout creating these environments, I explored two different types of reinforcement learning algorithms:
1. Value-based: the agent formulates a value function that takes in a state and/or action and outputs the expected reward. This is especially useful for environments with discrete actions, such as Snake (can either move forward, turn left, or turn right) and Flappy Bird (flap or no flap). In my project, these value functions took the form of a set of neural networks that estimate both the relative value of an agent being in a specific state, and an agent taking a specific action given that state.
2. Policy-based: the agent formulates a policy, which is a probability distribution for all possible actions the agent could take in a given state. Through training, the agent optimizes this policy, increasing the likelihood of good actions in certain states and decreasing the likelihood of bad actions. This was especially useful when training the MuJoCo robot, as the actions in this case are not discrete, thus making it impossible to assign a value to every possible joint angle and motor velocity.

Some other features added are multithreaded telemetry recorders so the users can watch the latest completed episode readily available while the agent continues to train in the background, prioritized experience replay buffers to optimize selection of training data, and lots of tuning of the reward functions to make sure the robot does exactly what you want it to do; no more, no less.


## Why did I make this?

A few months back, I went on a YouTube spiral of AI Warehouse, a channel showcasing different agents learning to play games and compete against each other. I was fascinated by the vast possibilities of machine learning, and thus I wanted to understand reinforcement learning beyond simply using a prebuilt library. Building the environments, agents, reward systems, and renderers from scratch allowed me to understand how each part of the training pipeline affects an agent's behavior.

Flappy Bird and Snake are simple, intuitive games that provide a way to explore value-based learning in environments with very different observations and  reward structures, each with their own unique challenges: it was hard to teach the snake not to trap itself within its own body, and it was equally difficult to get the physics just right for Flappy Bird to be able to learn. The MuJoCo biped extended the project into continuous control, exploring how AI could be used in a real world application: robotics. Getting it to learn how to balance was already difficult, let alone teaching it to walk forward; this forced me to think deeply about shaping the reward function in order to disallow the robot from taking shortcuts, making unexpected movements, and walking too fast or too slow.

Overall, this project taught me a lot about the logic and theory behind reinforcement learning, and really made me think critically about all the factors that could influence an agent's behavior and success.

## How to use/install

The demo supports 64-bit Windows 10 and Windows 11. Users do not need Python, Docker, a virtual environment, or an internet connection.

1. Download the release ZIP `RLArcade-Windows.zip` and extract it. Confirm that `RLArcade.exe` and `BipedDemo.exe` are in the same folder.
2. Double-click `RLArcade.exe`.
3. If Windows SmartScreen appears, select **More info**, then **Run anyway**.
4. Wait for the application to open.
5. Enjoy! Select **Flappy Bird** or **Snake** to experiment with different environments. You can either select **Comparison** mode to watch the trained DQN and random baseline run side by side, or select **Live training** and click **Start training** to train a new DQN from scratch while recorded episodes play in the application. Click **Stop** to end after the current episode.
6. To watch the MuJoCo robot, click **Watch MuJoCo** to open the trained biped in the native MuJoCo viewer. To ensure that this works, `BipedDemo.exe` and `RLArcade.exe` need to be in the same folder so this button can find it (explained in step 1).

New live-training checkpoints are stored under `%LOCALAPPDATA%\RLArcade`. They do not overwrite the trained models bundled with the demo.

To run the executables directly from this repository, open the `dist` folder and launch `RLArcade.exe`. `BipedDemo.exe` can also be opened directly to launch the trained biped viewer.
