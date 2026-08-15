RL ARCADE - WINDOWS DEMO

Requirements
------------
- 64-bit Windows 10 or Windows 11
- No Python, Docker, dependencies, or internet connection required

How to launch
-------------
1. Extract the entire ZIP before running anything.
2. Keep RLArcade.exe and BipedDemo.exe together in this folder.
3. Double-click RLArcade.exe.
4. If Windows SmartScreen appears, click More info, then Run anyway.
5. Allow several seconds for the first launch while bundled files unpack.

How to use
----------
- Select Flappy Bird or Snake.
- Use Comparison mode to watch the trained DQN beside a random baseline.
- Use Live training mode, then click Start training, to watch newly trained
  episodes while learning continues in the background.
- Click Stop to stop DQN training after the current episode.
- Click Watch MuJoCo to open the trained biped simulation.

BipedDemo.exe can also be opened directly. Do not rename or separate the two
executables, because RLArcade.exe looks for BipedDemo.exe beside itself.

Live-training checkpoints are saved under:
%LOCALAPPDATA%\RLArcade

They do not overwrite the trained models bundled in RLArcade.exe.
