from cart_pole_agent import CartPoleAgent, Observation
import gymnasium as gym
from tqdm import tqdm

N_EPISODES = 5000
EPISODES = range(N_EPISODES)
RENDER_EVERY = -1
TQDM = False

def run_episode(env: gym.Env, agent: CartPoleAgent):
    observation: Observation = env.reset()[0]
    log_probs = []
    rewards = []

    done = False
    while not done:
        action, log_prob = agent.select_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        done = terminated or truncated

    return log_probs, rewards



training_env = gym.make("CartPole-v1")
human_env = gym.make("CartPole-v1", render_mode="human")

agent = CartPoleAgent()

if TQDM:
    iterable = tqdm(EPISODES)
else:
    iterable = EPISODES

for episode in iterable:
    if RENDER_EVERY >= 0 and episode % RENDER_EVERY == 0:
        log_probs, rewards = run_episode(human_env, agent)
    else:
        log_probs, rewards = run_episode(training_env, agent)
    loss = agent.update(log_probs, rewards)

    if not TQDM:
        print(f"episode {episode}  reward={sum(rewards)}  loss={loss}")

agent.save_torch("agent.pt")
# agent.save_json("agent.json")