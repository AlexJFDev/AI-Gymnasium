import gymnasium as gym

# create environment
env = gym.make("CartPole-v1", render_mode="human")

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
