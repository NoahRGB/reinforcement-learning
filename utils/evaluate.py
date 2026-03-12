from agents.agent import Agent
from environments.environment import Environment
from environments.gym_environment import GymEnvironment

def evaluate(agent: Agent, env: Environment, resume: bool):


    s = env.get_start_state()
    
    if not resume:
        agent.initialise(env.get_state_space(), env.get_action_space(), s, resume=False)

    agent.toggle_eval()

    env.reset()
    t = 0
    done = False
    total_reward = 0

    while not done:
        a = agent.run_policy(s, t)
        sprime, r, done = env.step(s, a)
        agent.update(s, sprime, a, r, done)
        s = sprime
        total_reward += r
        t += 1

    print(f"EVALUATION EPISODE: reward {total_reward}")

    agent.toggle_eval()
        
