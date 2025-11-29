# AI-Gymnasium
Trying out Farama Foundation Gymnasium

[Gymnasium](https://gymnasium.farama.org/)

## Car Racing
Based on [this tutorial](https://gymnasium.farama.org/introduction/basic_usage/)

In this simple environment, a racecar sits on a track. Technically, the goal is to drive around the track, but in this case, the agent just moves randomly.

The environment is held in the variable `env`. It has many different methods for observing and manipulating the environment.

With `env.action_space` and `env.observation_space` the action and observation spaces of the environment can be read. These are instances of a `Space` class, which can represent many different spaces. In this case, the action space represents a vector of 3 floats. The first float is from -1 to 1 and the second and third are 0 to 1. The observation space is a 96x96 RBG grid.

With wrappers, the environment can be modified indirectly. In this case, the environment is wrapped in `FlattenObservation`. This represents the environment's observation as a 1d array. Obviously, this is helpful for certain machine learning.

The simulation is started by calling `env.reset()`. Then, in a while loop random actions are taken until the simulation reaches a terminal state. Reward is tracked for logging, but nothing is done with it since we aren't training.

## Blackjack
Based on [this tutorial](https://gymnasium.farama.org/introduction/train_agent/)

In this example, I am building a Blackjack playing agent using Q-learning. The agent is a class called `BlackjackAgent`.

### BlackjackAgent
I made this class fairly similar to what is outlined in the tutorial. As mentioned, it uses Q-learning. I also added my own methods for saving and loading the agents as either a `.pkl` or a `.json` file. Obviously the `.pkl` makes more sense to use, but it is interesting to be able to look at the `.json`.

### Helpers
These are very close to what is in the tutorial. They test and visualize the agent.

### Training
The loop and setup are based on the tutorial.

### Demo
Watch the agent play blackjack.

### Make Table
This script gives you a rough visualization of the Blackjack agent with tables. Online you can find a chart of ideal blackjack moves and mine are close-ish. Two tables are created. One is for a soft hand (when the player has an ace) and one is for a hard hand, when the player has no ace. This is because in blackjack, the ace can be either a 1 or 11 allowing for more flexibility than normal.

In each table, column 1 represents the dealer having an ace. There is no 1 valued card in blackjack.

The hard table includes rows for 22 to 31. This is because these are possible in the observation state of a blackjack game. However, in practice, the agent would never train on them because they represent loosing the game. Therefore, the agent defaults to hitting in these impossible states.

## CartPole
CartPole-v1 is another simulation provided by Gymnasium. It consists of balancing a pole on a cart. It's action space consists of either 1 or 0 while its observation space is 4 floats. For this reason, the Q learning used for Blackjack won't work here. Instead, I am using PyTorch.

### CartPoleAgent
This class implements an agent with a somewhat similar interface to the BlackJackAgent.

### Training
This loop is similar to the training loop for blackjack. After training, reward and loss progress are charted.

### Demo
This demo will load an agent and run it repeatedly in human mode until you tell it to quit.
