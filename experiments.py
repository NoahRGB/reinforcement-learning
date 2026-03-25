import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # shuts tensorflow up

import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *
from environments import *
from agents.tabular import *
from agents.approximate import *

device = detect_torch_device()
writer = create_tensorboard_writer(comment="")
print(f"using device {device}")

# =============== environments =================

# env = AtariEnvironment("ALE/Pong-v5", render_mode="human")
# env = GymEnvironment("LunarLander-v3", False, render_mode=None)
# env = GymEnvironment("Acrobot-v1", False, render_mode=None)
env = GymEnvironment("CartPole-v1", False, render_mode=None)
# env = GymEnvironment("MountainCar-v0", False, render_mode=None)
# env = GymEnvironment("Taxi-v3", False, render_mode=None)
# env = GymEnvironment("FrozenLake-v1", False, is_slippery=True, render_mode=None)
# env = GymEnvironment("CliffWalking-v1", False, render_mode=None)
# env = MazeEnvironment()

# =============== agents =================
# agent = ConvDQNAgent(device, writer, lr=1e-4, 
#                   replay_memory_size=10000, replay_warmup_length=10000,
#                   minibatch_size=32, 
#                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=150000,
#                   C=1000, gamma=0.99,
                  # save_nn_path="./torch_models/pong/pong_checkpoint.pt")
                  # load_nn_path="./torch_models/pong/pong_checkpoint.pt")
                  # load_nn_path="./results/bundles/pong/pong1/pong_checkpoint.pt")

# agent = DQNAgent(device, writer, lr=0.001, replay_memory_size=10000, C=1000,
                #  minibatch_size=32, epsilon=0.9, gamma=0.99, decay_rate=0.99)

agent = TDLambdaAgent(lambd=0.8, alpha=0.0001, epsilon=1.0, gamma=0.99, decay_rate=0.9)

# agent = ReinforceBaselineAgent(device, writer, policy_lr=0.001, state_value_lr=0.01, gamma=0.99, normalise=True)
# agent = ReinforceAgent(device, writer, lr=0.001, gamma=0.99, normalise=False)
# agent = SemigradientSarsaAgent(device, writer, normalise=False, lr=0.001, epsilon=0.9, gamma=0.99, decay_rate=0.99)


# agent = OffPolicyNstepSarsaAgentTB(n=4, alpha=0.1, epsilon=0.1, gamma=0.99)
# agent = OffPolicyNstepSarsaAgentISR(n=4, alpha=0.1, epsilon=0.1, gamma=0.99, expected=False)
# agent = OnPolicyNstepSarsaAgent(n=4, alpha=lr, epsilon=0.9, gamma=0.99, expected=False, decay_rate=0.99)
# agent = SarsaAgent(alpha=0.1, epsilon=0.9, gamma=0.99, expected=False, decay_rate=0.99)
# agent = QLearningAgent(alpha=0.1, epsilon=0.9, gamma=0.99, decay_rate=0.99)
# agent = OnPolicyMonteCarloAgent(epsilon=0.7, gamma=0.99, every_visit=False, decay_rate=0.999)




# =============== evaluating =================

# evaluate(agent, env, resume=False)

# =============== learning ==================

start = time.perf_counter()

episode_count = 1000
learning_rewards = learn(episode_count, env, agent, eval_period=0, quiet=False)

finished = time.perf_counter()
print(f"Finished in {round(finished - start, 2)} seconds")

plt.plot(learning_rewards)
plt.show()




writer.close()
