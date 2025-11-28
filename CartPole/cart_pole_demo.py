from cart_pole_agent import CartPoleAgent, Observation
import gymnasium as gym

AGENT_PATH = "agent.pt"

def run_episode(env: gym.Env, agent: CartPoleAgent):
    observation: Observation = env.reset()[0]

    done = False
    while not done:
        action, _log_prob = agent.select_action(observation)
        observation, _reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

env = gym.make("CartPole-v1", render_mode="human")

agent = CartPoleAgent()
agent.load_torch(AGENT_PATH)

print("Enter 'q' when you are ready to quit")
while input() != "q":
    run_episode(env, agent)
