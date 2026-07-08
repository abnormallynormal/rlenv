"""
Headless diagnostic: run the CURRENT deterministic policy (actor mean, no
sampling) and print WHY it scores the way it does. Answers the two questions
you can't get from watching:

  1. Is it actually moving forward (+X) or backward (-X)?  -> net displacement Δx
  2. Which reward term is it banking?                      -> summed reward terms
  3. Are the legs really locked?                           -> mean knee/hip angles

Run:  python policy/diagnose.py
"""
import numpy as np
import torch
from environment import Environment
from agent import Agent

CKPT = "agent_checkpoint.pt"
EPISODES = 5

env = Environment(xml="policy/robot.xml", height=0.95, max_steps=2000)
agent = Agent(state_size=16, action_dim=6, env=env)
try:
    agent.load(CKPT)
    print(f"loaded {CKPT} (observations={agent.observations})")
except Exception as e:
    print(f"could not load {CKPT} ({e}); running an UNTRAINED policy")

for ep in range(EPISODES):
    state = env.reset()
    start_x = float(env.data.qpos[0])
    totals = {}
    vels, r_knees, l_knees, r_hips, l_hips = [], [], [], [], []
    steps = 0
    terminated = False
    while True:
        st = torch.tensor(agent.normalize(state), dtype=torch.float32)
        with torch.no_grad():
            mean, _ = agent.actor(st)          # deterministic: intended behavior
        state, reward, done, terminated, _ = env.step(mean.numpy())
        steps += 1

        vels.append(float(state[15]))          # forward velocity (qvel[0])
        r_hips.append(float(state[0]))         # right hip angle
        r_knees.append(float(state[1]))        # right knee angle (0 = locked straight)
        l_hips.append(float(state[3]))         # left hip angle
        l_knees.append(float(state[4]))        # left knee angle

        if terminated:
            totals["terminal"] = totals.get("terminal", 0.0) - 10.0
        else:
            for k, v in env.reward_terms.items():
                totals[k] = totals.get(k, 0.0) + v

        if done:
            break

    dx = float(env.data.qpos[0]) - start_x
    direction = "FORWARD +X (good)" if dx > 0.05 else "BACKWARD -X (wrong way!)" if dx < -0.05 else "no net movement"
    tag = "TERMINATED (fell / tipped past 46 deg)" if terminated else "survived full episode"

    print(f"\n--- episode {ep}  |  {steps} steps  |  {tag} ---")
    print(f"  net displacement Δx : {dx:+.3f} m   -> {direction}")
    print(f"  mean fwd velocity   : {np.mean(vels):+.3f} m/s")
    print(f"  mean hip  R / L     : {np.mean(r_hips):+.3f} / {np.mean(l_hips):+.3f} rad")
    print(f"  mean knee R / L     : {np.mean(r_knees):+.3f} / {np.mean(l_knees):+.3f} rad  (0.0 = locked straight)")
    print(f"  reward terms (summed over episode):")
    for k in ["forward", "alive", "upright", "control", "smoothness", "terminal"]:
        if k in totals:
            print(f"      {k:11s}: {totals[k]:+8.1f}")
