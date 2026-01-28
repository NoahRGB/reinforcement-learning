# Reinforcement Learning
Implementations of various RL algorithms

`learn.py` contains logic for running any of the agents in any of the environments
`experiments.py` is used to try out different agents in different environments

## `agents/`
Contains implementations of various RL algorithms following the `agents/agent.py` base class description

These are used with the functions in `learn.py` to solve the environments from the `environments/` folder

`tabular` contains age

Tabular based agents:
- [x] On policy Monte Carlo
- [x] Q-learning
- [x] SARSA(0)
- [x] n-step SARSA (+ expected n-step SARSA)
- [x] Off policy n-step SARSA (using importance sampling + tree backup)
- [ ] Off policy Monte Carlo
- [ ] Q($\sigma$) off policy n-step SARSA

Approximate agents:
- [x] Semigradient SARSA(0) (using NN for q(s, a) approximation)
- [ ] REINFORCE

## `environments/`
Contains a basic maze environment as well as support for different [Farama Gymnasium](https://gymnasium.farama.org/) environments. These can be used with the functions in `learn.py`

## `results/` and `graph_data/` and `torch_models/`
Contains any meaningful graphs/plots/videos generated in `experiments.py` and the data used to generate them (in .pkl format)

## `archive/maze_learning/`
Contains a simple maze environment with a start state + goal state. Includes a pygame script for displaying the agents as they try to solve the maze. The agents from this have been improved and refactored into `agents/` and the maze environment (without the pygame aspects) is available in `environments/` 

## `archive/bandits/`
Contains some experiments based on K-armed bandits
Follows the concepts as explained in [Sutton & Barto Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

