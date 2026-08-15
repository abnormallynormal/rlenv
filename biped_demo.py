"""Train or watch the project's original MuJoCo PPO biped."""
import argparse, time
from pathlib import Path

import mujoco
import mujoco.viewer
import torch

from arcade.resources import resource_path
from policy.agent import Agent
from policy.environment import Environment


def load(agent, checkpoint, modified=0.):
    try:
        stamp = checkpoint.stat().st_mtime
        if stamp != modified:
            agent.load(checkpoint)
            if agent.observations > 0: return stamp
    except (OSError, RuntimeError, EOFError, ValueError, ZeroDivisionError): pass
    return modified


def watch(checkpoint, smoke=False):
    env = Environment(str(resource_path("policy","robot.xml")),.95,max_steps=2000)
    agent = Agent(state_size=16,action_dim=6,env=env); modified = load(agent,checkpoint)
    if agent.observations <= 0:
        checkpoint=resource_path("agent_checkpoint.pt"); modified=load(agent,checkpoint)
    if agent.observations <= 0: raise RuntimeError("No initialized PPO checkpoint is available.")
    state = env.reset()
    if smoke:
        for _ in range(25):
            with torch.no_grad(): action,_ = agent.actor(torch.tensor(agent.normalize(state),dtype=torch.float32))
            state,*_ = env.step(action.numpy())
        print("MuJoCo smoke test passed"); return
    with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
        while viewer.is_running():
            started=time.perf_counter()
            with torch.no_grad(): action,_ = agent.actor(torch.tensor(agent.normalize(state),dtype=torch.float32))
            state,_reward,done,_terminated,_ = env.step(action.numpy()); viewer.sync()
            if done: state=env.reset(); modified=load(agent,checkpoint,modified)
            time.sleep(max(0.,env.model.opt.timestep-(time.perf_counter()-started)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,default=resource_path("agent_checkpoint.pt"))
    parser.add_argument("--smoke-test",action="store_true")
    args=parser.parse_args(); watch(args.checkpoint.resolve(),args.smoke_test)


if __name__=="__main__": main()
