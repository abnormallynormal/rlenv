from train import train_loop
from agent.buffer import Buffer
from modules.snake.environment import Environment as snake
from modules.flappy.environment import Environment as flappy
from modules.snake.telemetry import TelemetryRecorder as snake_recorder
from modules.flappy.telemetry import TelemetryRecorder as flappy_recorder
from modules.snake.renderer import Renderer as snake_renderer
from modules.flappy.renderer import Renderer as flappy_renderer
from agent.dqn import Agent as dqn
from collections import deque
import threading, time

def training(environment, agent, replay_buffer, telemetry, action_size):
  epsilon = 0.9999
  max_score = 0
  N = 0
  for episode in range(12000):
    N, epsilon, total_reward, score = train_loop(environment, agent, replay_buffer, epsilon, action_size, N, telemetry)
    epsilon = max(0.999 * epsilon, 0.01)
    max_score = max(max_score, score)
    print(f"Episode {episode} done, epsilon: {epsilon:.3f}, reward: {total_reward:.3f}, score: {score}, max: {max_score}")
  print(f"Max score: {max_score}")
def main():
  VISUALIZE = True
  
  state_size = 4
  action_size = 2
  environment = flappy()
  agent = dqn(state_size, action_size)
  replay_buffer = Buffer(alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5)

  if VISUALIZE:
    renderer_ready = {"ready": True}
    mutex = threading.Lock()
    
    renderer = flappy_renderer(400, 600)
    playback_queue = deque()
    telemetry = flappy_recorder(playback_queue, renderer_ready, mutex)
    
    training_thread = threading.Thread(target=training, args=(environment, agent, replay_buffer, telemetry, action_size), daemon=True)
    training_thread.start()
    
    while True:
      if playback_queue:
        with mutex:
          renderer_ready["ready"] = False
        recorded_episode = playback_queue.popleft()
        for frame in recorded_episode:
          renderer.render_frame(frame)
        with mutex:
          renderer_ready["ready"] = True
      else:
        time.sleep(0.01)
  else:
    print("Visualization disabled. Training running at maximum hardware capacity...")
    epsilon = 0.9999
    
    for episode in range(5000): 
      N = 0
      epsilon, total_reward = train_loop(environment, agent, replay_buffer, epsilon, action_size, N, recorder=None)
      epsilon *= 0.99
      print(f"Episode {episode} done, epsilon: {epsilon:.3f}, reward: {total_reward:.3f}")

if __name__ == "__main__":
  main()