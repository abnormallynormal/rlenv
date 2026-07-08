import torch
from environment import Environment   # match your actual import
from agent import Agent               # match your actual import
import mujoco
import mujoco.viewer
import time
import os

CKPT = "agent_checkpoint.pt"

# Rebuild the same env/agent structure you trained with
env = Environment(xml="policy/robot.xml", height=0.95, max_steps=2000)  # match your real constructor args
agent = Agent(state_size=16, action_dim=6, env=env)  # match your real state_size


def try_reload(last_mtime):
    """Reload the checkpoint only if it changed on disk. Survives the file not
    existing yet and main.py being mid-save. Returns the mtime we're now on."""
    try:
        mtime = os.path.getmtime(CKPT)
    except OSError:
        return last_mtime            # main.py hasn't written a checkpoint yet
    if mtime == last_mtime:
        return last_mtime            # unchanged since last load
    try:
        agent.load(CKPT)
        print(f"[watch] reloaded checkpoint (observations={agent.observations})", flush=True)
        return mtime
    except Exception:
        return last_mtime            # caught a mid-write; try again next time

last_mtime = try_reload(0.0)         # load once at startup (ok if it's not there yet)
state = env.reset()

SLOWDOWN = 1.0  # 1.0 = real time; 2.0 = half speed; 4.0 = quarter speed, etc.

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        state_tensor = torch.tensor(agent.normalize(state), dtype=torch.float32)
        with torch.no_grad():
            mean, log_std = agent.actor(state_tensor)
            action = mean  # use the mean directly, not a random sample, to see its "intended" behavior

        state, reward, done, terminated, _ = env.step(action.numpy())

        viewer.sync()

        if done:
            state = env.reset()
            last_mtime = try_reload(last_mtime)  # pick up the newest policy between episodes

        # pace the loop so sim-time tracks wall-clock (otherwise it runs ~hundreds of x too fast)
        wait = env.model.opt.timestep * SLOWDOWN - (time.time() - step_start)
        if wait > 0:
            time.sleep(wait)