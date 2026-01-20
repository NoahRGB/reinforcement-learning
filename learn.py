import matplotlib.pyplot as plt

from environments.gym_environment import GymEnvironment
from environments.maze_environment import MazeEnvironment

from agents.qlearning_agent import QLearningAgent
from agents.montecarlo_agent import MonteCarloAgent

episode_count = 50

env = GymEnvironment("CliffWalking-v1", None)
# env = MazeEnvironment()

# agent = QLearningAgent(alpha=0.5, epsilon=0.5, gamma=0.99, decay_rate=0.99)
agent = MonteCarloAgent(epsilon=0.9, gamma=0.99, decay_rate=0.99)

agent.initialise(env.get_state_space_size(), env.get_action_space_size())

for episode_num in range(episode_count):
    env.reset()
    s = env.get_start_state()
    done = False
    while not done:
        a = agent.run_policy(s)
        sprime, r, done = env.step(s, a)
        agent.update(s, sprime, a, r)
        s = sprime
    agent.finish_episode()

plt.plot([reward*-1 for reward in agent.reward_history])
plt.show()
