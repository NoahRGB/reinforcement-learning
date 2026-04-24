From scratch implementations of various reinforcement learning algorithms using NumPy and PyTorch
## Algorithms implemented

- [x] Semigradient SARSA
- [x] REINFORCE
- [x] DQN
- [x] A2C
- [x] PPO
- [ ] TRPO

<br/>

- [x] Tabular on/off policy (nstep) SARSA
- [x] Tabular Q-learning
- [x] Tabular Monte Carlo


## [Classic control](https://gymnasium.farama.org/environments/classic_control/) environments
| Environment       | DQN | A2C | REINFORCE | Semigradient SARSA | PPO |
|------------------|:---:|:---:|:---------:|:------------------:|:---:|
| Acrobot          |  ✓  |  ✓  |          |                   |    |
| CartPole         |  ✓  |  ✓  |     ✓     |         ✓          |    |
| Mountain Car     |     |    |          |                   |    |
| Pendulum         |     |    |          |                   |    |


## [Box2D (physics control)](https://gymnasium.farama.org/environments/box2d/) environments
| Environment       | DQN | A2C | REINFORCE | Semigradient SARSA | PPO |
|------------------|:---:|:---:|:---------:|:------------------:|:---:|
| Bipedal Walker   |     |    |          |                   |    |
| Car Racing       |  ✓  |  ✓  |          |                   |    |
| Lunar Lander     |  ✓  |    |          |                   |    |

## [MuJoCo (Multi-Joint dynamics with Contact)](https://gymnasium.farama.org/environments/mujoco/) environments
| Environment            | A2C | REINFORCE | PPO |
| ---------------------- | :-: | :-------: | :-: |
| Ant                    |     |           |     |
| HalfCheetah            |     |           |     |
| Hopper                 |     |           |     |
| Humanoid               |     |           |     |
| HumanoidStandup        |     |           |     |
| InvertedDoublePendulum |     |           |     |
| InvertedPendulum       |     |           |     |
| Pusher                 |     |           |     |
| Reacher                |     |           |     |
| Swimmer                |     |           |     |
| Walker2D               |     |           |     |


## [Atari](https://ale.farama.org/environments/) environments
| Environment    | DQN | A2C | PPO |
| -------------- | :-: | :-: | :-: |
| Pong           |  ✓  |     |     |
| Space Invaders |  ✓  |     |     |

<br/>

## Examples

![DQN pong agent](./results/finished/dqn/dqn_pong/dqn_pong_master.gif)


![DQN car racing agent](./results/finished/dqn/dqn_carracing/dqn_carracing.gif)

![DQN boxing agent](./results/finished/dqn/dqn_boxing/dqn_boxing.gif)