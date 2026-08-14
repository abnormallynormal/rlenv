"""Standalone live viewer for the trained MuJoCo PPO biped."""

import math
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch import nn

from arcade.resources import resource_path


class Actor(nn.Module):
    def __init__(self, state_size=16, action_dim=6):
        super().__init__()
        self.hidden_1 = nn.Linear(state_size, 64)
        self.hidden_2 = nn.Linear(64, 64)
        self.distributions = nn.Linear(64, action_dim * 2)

    def forward(self, inputs):
        values = torch.tanh(self.hidden_1(inputs))
        values = torch.tanh(self.hidden_2(values))
        means, log_stds = self.distributions(values).chunk(2, dim=-1)
        return means, torch.clamp(log_stds, -20, 2)


def observation(data):
    positions = data.qpos
    velocities = data.qvel
    return np.concatenate(
        [
            positions[3:9],
            velocities[3:9],
            [positions[1], velocities[1], data.xpos[1][2], velocities[0]],
        ]
    )


def reset(model, data):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return observation(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="load assets and step without opening a viewer")
    args = parser.parse_args()
    xml_path = resource_path("policy", "robot.xml")
    checkpoint_path = resource_path("agent_checkpoint.pt")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor = Actor()
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    count = max(1, int(checkpoint["observations"]))
    variance_sum = np.asarray(checkpoint["sum_of_sd"], dtype=np.float32)
    deviation = np.sqrt(variance_sum / count + 1e-8)

    state = reset(model, data)
    if args.smoke_test:
        for _ in range(25):
            normalized = (state - mean) / deviation
            with torch.no_grad():
                action, _log_std = actor(torch.as_tensor(normalized, dtype=torch.float32))
            data.ctrl[:] = action.numpy()
            mujoco.mj_step(model, data)
            state = observation(data)
        print(f"MuJoCo smoke test passed: observation={state.shape}, controls={model.nu}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            normalized = (state - mean) / deviation
            with torch.no_grad():
                action, _log_std = actor(torch.as_tensor(normalized, dtype=torch.float32))
            data.ctrl[:] = action.numpy()
            mujoco.mj_step(model, data)
            state = observation(data)
            fallen = state[14] < 0.6 or abs(state[12]) > 0.8
            if fallen:
                state = reset(model, data)
            viewer.sync()
            delay = model.opt.timestep - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
