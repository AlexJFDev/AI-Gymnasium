from blackjack_agent import BlackjackAgent
import gymnasium as gym

env = gym.make("Blackjack-v1", sab=False, render_mode="human")

agent = BlackjackAgent(env, 0, 0, 0, 0)

agent.load_pickle("agent.pkl")

observation, info = env.reset()
done = False
while not done:
    action = agent.get_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
