import numpy as np
import mujoco
class Environment:
  def __init__(self, xml, height, max_steps=2000):
    self.model = mujoco.MjModel.from_xml_path(xml)
    self.data = mujoco.MjData(self.model)
    self.height = height
    self.max_steps = max_steps
    self.steps = 0
    self.prev_ctrl = np.zeros(self.model.nu)
    mujoco.mj_resetData(self.model, self.data)
  def reset(self):
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:] += np.random.uniform(-0.01, 0.01, self.model.nq)
    self.data.qvel[:] += np.random.uniform(-0.01, 0.01, self.model.nv)
    self.steps = 0
    self.prev_ctrl = np.zeros(self.model.nu)

    mujoco.mj_forward(self.model, self.data)

    return self.get_state()
  def step(self, action):
    self.data.ctrl[:] = action
    mujoco.mj_step(self.model, self.data)
    self.steps += 1
    
    state = self.get_state()
    terminated = self.is_done(state)
    truncated = self.steps >= self.max_steps
    is_done = terminated or truncated
    reward = -10 if terminated else self.calculate_reward(state)
    self.prev_ctrl = self.data.ctrl.copy()   # remember torques for next step's smoothness term
    return (state, reward, is_done, terminated, {})
  def get_state(self):
    positions = self.data.qpos
    velocities = self.data.qvel
                                   
    torso_height = self.data.xpos[1][2]
    torso_pitch = positions[1]
    torso_pitch_velocity = velocities[1]
    
    leg_joint_angles = positions[3:9]
    leg_joint_velocities = velocities[3:9]
    
    velocity = velocities[0]
    
    return np.concatenate([leg_joint_angles, leg_joint_velocities, [torso_pitch], [torso_pitch_velocity], [torso_height], [velocity]])
  
  
  def is_done(self, state):
    torso_height = state[14]
    torso_pitch = state[12]
    return torso_height < 0.6 or abs(torso_pitch) > 0.8
    
  def calculate_reward(self, state):
    forward_velocity = state[15]
    torso_pitch = state[12]

    # Velocity tracking: reward peaks at the target speed and falls off if the
    # robot goes too SLOW or too FAST. The old min(v, target) cap let it rush
    # (you measured ~1.7 m/s) and outrun its own balance; tracking pulls it to
    # a controlled pace, which is steadier and looks more natural.
    target_speed = 1.0
    forward_reward = 1.5 * np.exp(-2.0 * (forward_velocity - target_speed) ** 2)

    # Stay alive and roughly upright. Small on purpose: it must never
    # outweigh forward progress, or standing still becomes optimal again.
    alive_bonus = 0.5
    upright_penalty = 1.0 * (torso_pitch ** 2)

    # Light energy/control cost. This discourages flailing WITHOUT punishing
    # the leg swing that walking is made of (which the old leg-velocity term
    # did). self.data.ctrl holds the torques actually applied this step.
    control_cost = 0.001 * np.sum(self.data.ctrl ** 2)

    # Smoothness: penalize sudden torque changes. This is what kills the
    # "violent jerk" — a smooth gait reverses torques gradually, a jerk slams
    # them, so (ctrl - prev_ctrl)^2 spikes on the jerk but stays small when
    # walking. Dial this DOWN if the gait comes out too timid/stiff.
    smoothness_cost = 0.05 * np.sum((self.data.ctrl - self.prev_ctrl) ** 2)

    # Signed contribution of each term, so a diagnostic can see which one the
    # policy is actually banking (they must sum to the returned reward).
    self.reward_terms = {
      "forward": forward_reward,
      "alive": alive_bonus,
      "upright": -upright_penalty,
      "control": -control_cost,
      "smoothness": -smoothness_cost,
    }

    return alive_bonus + forward_reward - upright_penalty - control_cost - smoothness_cost
  