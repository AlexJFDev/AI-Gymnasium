# AI-Gymnasium
Trying out Farama Foundation Gymnasium

[Gymnasium](https://gymnasium.farama.org/)

## Car Racing
Based on [this tutorial](https://gymnasium.farama.org/introduction/basic_usage/)

In this simple environment, a racecar sits on a track. Technically, the goal is to drive around the track, but in this case, the agent just moves randomly.

The environment is held in the variable `env`. It has many different methods for observing and manipulating the environment.

With `env.action_space` and `env.observation_space` the action and observation spaces of the environment can be observed. These are instances of a `Space` class, which can represent many different spaces. In this case, the action space represents a vector of 3 floats. The first float is from -1 to 1 and the second and third are 0 to 1. The observation space is a 96x96 RBG grid.

With wrappers, the environment can be modified indirectly. In this case, the environment is wrapped in `FlattenObservation`. This represents the environment's observation as a 1d array. Obviously, this is helpful for certain machine learning.

The simulation is started by calling `env.reset()`. Then, in a while loop random actions are taken until the simulation reaches a terminal state. Reward is tracked for logging, but nothing is done with it yet.