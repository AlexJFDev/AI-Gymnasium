from cart_pole_agent import CartPoleAgent, Observation
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import axes

N_EPISODES = 500
EPISODES = range(N_EPISODES)
RENDER_EVERY = 100
TQDM = True

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

def plot(total_rewards: list[float], losses: list[float]):
    fig, plots = plt.subplots(ncols=2, figsize=(12, 5))

    reward_plot: axes.Axes = plots[0]
    reward_plot.set_title("Total Rewards")
    reward_plot.set_ylabel("Total Reward")
    reward_plot.set_xlabel("Iteration")
    reward_plot.plot(
        range(len(total_rewards)),
        total_rewards
    )

    loss_plot: axes.Axes = plots[1]
    loss_plot.set_title("Losses")
    loss_plot.set_ylabel("Loss")
    loss_plot.set_ylabel("Iteration")
    loss_plot.plot(
        range(len(losses)),
        losses
    )

    plt.tight_layout()
    plt.show()



training_env = gym.make("CartPole-v1")
human_env = gym.make("CartPole-v1", render_mode="human")

agent = CartPoleAgent()

if TQDM:
    iterable = tqdm(EPISODES)
else:
    iterable = EPISODES

total_rewards = []
losses = []
for episode in iterable:
    if RENDER_EVERY >= 0 and episode % RENDER_EVERY == 0:
        run_episode(human_env, agent)
    
    log_probs, rewards = run_episode(training_env, agent)
    loss = agent.update(log_probs, rewards)

    total_reward = sum(rewards)

    total_rewards.append(total_reward)
    losses.append(loss)

    if not TQDM:
        print(f"episode {episode}  reward={total_reward}  loss={loss}")

plot(total_rewards, losses)

agent.save_torch("agent.pt")
# agent.save_json("agent.json")