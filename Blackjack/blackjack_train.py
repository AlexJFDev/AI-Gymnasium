from blackjack_agent import BlackjackAgent
import gymnasium as gym
from tqdm import tqdm
import helpers

learning_rate = .01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = .1

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env,
    learning_rate,
    start_epsilon,
    epsilon_decay,
    final_epsilon
)

for episode in tqdm(range(n_episodes)):
    # Equivalent to dealing a fresh hand
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)

        agent.update(observation, action, reward, terminated, next_observation)

        done = terminated or truncated
        observation = next_observation
    
    agent.decay_epsilon()

helpers.visualize_agent(agent, env)
helpers.test_agent(agent, env)

agent.save_json("agent.json")
agent.save_pickle("agent.pkl")
