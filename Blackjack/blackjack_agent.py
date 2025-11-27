from collections import defaultdict
import gymnasium as gym
import numpy as np
import pickle
import json

Observation = tuple[int, int, bool]

class BlackjackAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = .095
        ):
        self.env = env

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, observation: Observation):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        return int(np.argmax(self.q_values[observation]))
    
    def update(
            self,
            observation: Observation,
            action: int,
            reward: float,
            terminated: bool,
            next_observation: Observation
        ):
            future_q = (not terminated) * np.max(self.q_values[next_observation])

            target = reward + self.discount_factor * future_q
            
            temporal_difference = target - self.q_values[observation][action]

            self.q_values[observation][action] = (
                self.q_values[observation][action] + self.learning_rate * temporal_difference
            )

            self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: str, readable=False):
        if readable:
            self.save_json(path)
        else:
            self.save_pickle(path)
    
    def save_json(self, path: str):
        data = {}
        for state, values in self.q_values.items():
            key = str(state)
            data[key] = values.tolist()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_values), f)

    def load(self, path: str, readable=False):
        if readable:
            self.load_json(path)
        else:
            self.load_pickle(path)


    def load_json(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        
        self.q_values = defaultdict(
            lambda: np.zeros(self.env.action_space.n)
        )

        for key, values in data.items():
            state = eval(key)
            self.q_values[state] = np.array(values, dtype=float)

    def load_pickle(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_values = defaultdict(
            lambda: np.zeros(self.env.action_space.n),
            data
        )