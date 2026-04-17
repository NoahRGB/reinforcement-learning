# Reinforcement Learning
Implementations of various RL algorithms

`experiments.py` is where the current agent/environment is imported and learning is started


`agents/` contains the base class `Agent` as well as various tabular or approximate implementations of algorithms as subclasses `agents/approximate/`, `agents/tabular/`


`environments/` contains the base class `Environment` as well as various implementations of different environments as subclasses (e.g. `environments/gym_environment.py`)


`utils/` contains files imported by `experiments.py` that facilitate agent-environment interactions (e.g. `utils/learn.py`)


`archive/` contains older experiments like K-armed bandits (`archive/bandits/`) or maze-based learning (`archive/mazelearning/`)


`results/`, `graph_data/`, `torch_models/` contain various saved experiments for visualising


![DQN pong agent](./results/finished/dqn/dqn_pong/dqn_pong_master.gif)