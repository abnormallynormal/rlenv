import numpy as np
import random
import mujoco
class Environment:
  def __init__(self, xml, height):
    self.model = mujoco.MjModel.from_xml_path(xml)
    self.data = mujoco.MjData(self.model)
    self.height = height
    mujoco.mj_resetData(self.model, self.data)
  def reset(self):
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:] += np.random.uniform(-0.01, 0.01, self.model.nq)
    self.data.qvel[:] += np.random.uniform(-0.01, 0.01, self.model.nv)
    
    mujoco.mj_forward(self.model, self.data)

    return self.get_state()
  def step(self, action):
    self.data.ctrl[:] = action
    mujoco.mj_step(self.model, self.data)
    
    state = self.get_state()
    is_done = self.is_done(state)
    reward = -2 if is_done else self.calculate_reward(state)
    return (state, reward, is_done, {})
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
    return torso_height < 0.5
    
  def calculate_reward(self, state):
    velocity = state[15]
    leg_joint_velocities = state[6:12]
    torso_pitch = state[12]
    torso_pitch_velocity = state[13]
    torso_height = state[14]

    
    return (velocity * 1.0 - np.sqrt(np.mean(leg_joint_velocities ** 2)) * 0.2 - abs(torso_pitch_velocity) * 0.3 - abs(torso_pitch) - (self.height - torso_height) * 2.0 + 0.5)
  