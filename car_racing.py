import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# create environment
env = gym.make("CarRacing-v3", render_mode="human")

print(f"Action space: {env.action_space}")
# Complicated, basically it is a vector of 3 floats. The first from -1 to 1 and the second and third from 0 to 1.
print(f"Random Action: {env.action_space.sample()}")
# 3 random floats in the range.

print(f"Observation space: {env.observation_space}") # Box with 4 values
# Representing the entire 96x96 RGB environment.

print()
print()
print()

print(env.observation_space.shape) # 96x96 RBG
flattened_env = FlattenObservation(env)
print(flattened_env.observation_space.shape) # 27648

print()
print()
print()

# Get initial observation
observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    # Pick a random action
    action = env.action_space.sample()
    # Take the action
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
