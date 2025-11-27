from blackjack_agent import BlackjackAgent
import gymnasium as gym

ACTION_MAP = {
    0: "stand",
    1: "hit"
}

OBSERVATION_MAP = {
    -1: "loose",
    0: "continue",
    1: "win"
}

env = gym.make("Blackjack-v1", sab=False, render_mode="human")

agent = BlackjackAgent(env, 0, 0, 0, 0)

agent.load_pickle("agent_trained.pkl")

observation, info = env.reset()
done = False
while not done:
    agent_value, dealer_value, agent_has_ace = observation
    ace_text = "an ace" if agent_has_ace else "no ace"
    print(f"Agent has {agent_value} and {ace_text}, dealer has {dealer_value}")
    input("Hit enter to continue")
    action = agent.get_action(observation)
    print(f"Agent chose to {ACTION_MAP[action]}")
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(OBSERVATION_MAP[reward])
    print()
    print()