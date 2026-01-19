import matplotlib.pyplot as plt

from environments.gym_environment import GymEnvironment
from environments.maze_environment import MazeEnvironment
from agents.qlearning_agent import QLearningAgent

episode_count = 2000
env = GymEnvironment("Taxi-v3", None)
# env = MazeEnvironment()
agent = QLearningAgent(alpha=0.5, epsilon=0.5, gamma=0.99, decay=True)
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

plt.plot([reward for reward in agent.reward_history])
plt.show()
