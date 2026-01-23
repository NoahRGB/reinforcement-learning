# Reinforcement Learning
Implementations of various RL algorithms

`learn.py` contains logic for running any of the agents in any of the environments. `experiments.py` is used to try out different agents in different environments

#### `agents/`
Contains implementations of various RL algorithms following the `agents/agent.py` base class description. These are used with the functions in `learn.py` to solve the environments from the `environments/` folder

Tabular based agents:
- [ ] On policy Monte Carlo
- [ ] Q-learning
- [ ] Sarsa(0)
- [ ] n-step Sarsa (+ expected n-step Sarsa)
- [ ] Off policy n-step Sarsa (using importance sampling + tree backup)

#### `environments/`
Contains a basic maze environment as well as support for different [Farama Gymnasium](https://gymnasium.farama.org/) environments. These can be used with the functions in `learn.py`

#### `results/` and `graph_data/`
Contains any meaningful graphs/plots generated in `experiments.py` and the data used to generate them (in .pkl format)

#### `archive/maze_learning/`
Contains a simple maze environment with a start state + goal state. Includes a pygame script for displaying the agents as they try to solve the maze. The agents from this have been improved and refactored into `agents/` and the maze environment (without the pygame aspects) is available in `environments/` 

#### `archive/bandits/`
Contains some experiments based on K-armed bandits
Follows the concepts as explained in [Sutton & Barto Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

