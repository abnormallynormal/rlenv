import numpy as np
import mujoco
import mujoco_viewer
from environment import Environment
from agent import Agent
import os

# model = mujoco.MjModel.from_xml_path("policy/robot.xml")
# data = mujoco.MjData(model)

# mujoco.mj_resetData(model, data)
# mujoco.mj_forward(model, data)

# print("torso world pos at start:", data.xpos[1])
# print("qpos[2] at start:", data.qpos[2])

# viewer = mujoco_viewer.MujocoViewer(model, data)

# for _ in range(10000):
#     if viewer.is_alive:
#         mujoco.mj_step(model, data)
#         viewer.render()
#         print("velocity:", np.mean(data.qvel[3:9]))
#     else:
#         break

# viewer.close()

env = Environment("policy/robot.xml", 0.95)
agent = Agent(env=env, state_size=16, action_dim=6)

# Resume from the last checkpoint instead of starting from scratch.
# Set RESUME = False (or delete the file) to train a fresh policy.
RESUME = True
CKPT = "agent_checkpoint.pt"
if RESUME and os.path.exists(CKPT):
    agent.load(CKPT)
    print(f"resuming from {CKPT} (observations={agent.observations})")
else:
    print("starting from scratch")

agent.train(iterations=1000, gamma=0.99, LAMBDA=0.95, epsilon=0.2, c2=0.005)

