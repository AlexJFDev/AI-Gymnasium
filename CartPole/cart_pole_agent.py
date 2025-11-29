# Parts of this file were made with ChatGPT, but I put them together into a class myself and added documentation.

import torch
import torch.nn as nn
import torch.distributions as D
import torch.optim as optim

Observation = tuple[float, float, float, float]

GAMMA_KEY = "gamma"
POLICY_KEY = "policy_state"
OPTIMIZER_KEY = "optimizer_state"

def normalize(x: torch.Tensor):
    return (x - x.mean()) / (x.std() + 1e-8)

class CartPolePolicyNetwork(nn.Module):
    """
    Policy Network for CartPoleAgent with 4 inputs, 64 hidden, and 2 outputs.
    """
    def __init__(
            self,
            
        ):
        super().__init__()
        # This makes a network with 4 inputs, 64 hidden nodes, and 2 outputs. It uses reinforcement learning.
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x): # Forward is used as the `__call__` method of `nn-Module`
        # Get output from the network
        return torch.softmax(self.model(x), dim=-1)
    
class CartPoleAgent:
    """
    Agent class for the CartPole game. Roughly similar interface to the BlackJackAgent
    """
    def __init__(
            self, 
            gamma=0.99, 
            optimizer_type: optim.Optimizer=optim.Adam, 
            learning_rate=0.01
        ):
        self.policy = CartPolePolicyNetwork()
        self.optimizer: optim.Adam = optimizer_type(
            self.policy.parameters(),
            lr=learning_rate
        )
        self.gamma = gamma
    
    def select_action(self, observation: Observation):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        probabilities = self.policy(observation_tensor)
        distribution = D.Categorical(probabilities)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def update(
        self,
        log_probabilities: list,
        rewards: list
    ):
        returns = self.compute_returns(rewards)
        return self.update_policy(log_probabilities, returns)

    def compute_returns(self, rewards: list):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)
    
    def update_policy(
        self,
        log_probabilities: list,
        returns: torch.Tensor
    ):
        returns = normalize(returns)
        loss = -(returns * torch.stack(log_probabilities)).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_torch(self, path):
        torch.save({
            POLICY_KEY: self.policy.state_dict(),
            OPTIMIZER_KEY: self.optimizer.state_dict(),
            GAMMA_KEY: self.gamma,
        }, path)

    def load_torch(self, path):
        checkpoint = torch.load(path)
        
        self.policy.load_state_dict(checkpoint[POLICY_KEY])
        self.optimizer.load_state_dict(checkpoint[OPTIMIZER_KEY])
        self.gamma = checkpoint[GAMMA_KEY]
