import random, math

from utils import argmax 

class Bandit:
    def __init__(self, k, step_size, q_centre=0):
        self.k = k
        self.step_size = step_size
        self.q_centre = q_centre
    
    def reset(self):
        self.N, self.q = {}, {}
        self.rewards = []
        for arm in range(0, self.k):
            self.N[arm] = 0
            self.q[arm] = random.gauss(self.q_centre, 1)
        self.optimal_action = argmax(self.q)
    
    def run(self):
        raise NotImplementedError("run not implemented")

    def update(self, action):
        raise NotImplementedError("update_Q not implemented")

    def calculate_optimal_action_count(self):
        # return sum([value for (key, value) in self.N.items() if key==self.optimal_action]) 
        return self.N[self.optimal_action]

    def get_config(self):
        raise NotImplementedError("get_config not implemented")

    def __str__(self):
        raise NotImplementedError("__str__ not implemented")

class ActionValueBandit(Bandit):
    def __init__(self, k, Q1, epsilon=None, c=None, step_size=None):
        super().__init__(k, step_size)
        self.epsilon = epsilon
        self.c = c
        self.Q1 = Q1
        self.reset()

    def reset(self):
        super().reset()
        self.Q = {}
        for arm in range(0, self.k):
            self.Q[arm] = [self.Q1]
   
    def run(self):
        if self.c != None and self.epsilon == None:
            action = argmax({key: (vals[-1] + self.c * math.sqrt(math.log(len(self.rewards)+1) / (self.N[key] if self.N[key] > 0 else 1))) for (key, vals) in self.Q.items()})
        elif self.epsilon != None and self.c == None:
            if self.epsilon and random.random() < self.epsilon: # epsilon% of the time, use a random action/arm
                action = random.randint(0, self.k-1)
            else: # otherwise use the action with the best Q
                action = argmax({key: vals[-1] for (key, vals) in self.Q.items()})
        else:
            print("Cannot have a UCB c and an epsilon")
            return -1

        self.N[action] += 1
        self.rewards.append(random.gauss(self.q[action], 1)) # reward is a normal distribution around the true reward
        self.update(action)
        return self.rewards[-1]

    def update(self, action):
        if not self.step_size: # no step size defined, so use sample average to update Q
            self.Q[action].append(self.Q[action][-1] + ((self.rewards[-1] - self.Q[action][-1]) / self.N[action]))
        else: # use step_size as the fixed step_size a to update Q
            self.Q[action].append(self.Q[action][-1] + self.step_size * (self.rewards[-1] - self.Q[action][-1]))

    def get_config(self):
        return f"k={self.k},ε={self.epsilon},c={self.c},Q1={self.Q1},a={self.step_size if self.step_size else '1/n'}"

    def __str__(self):
        string = f"{self.k}-armed action value bandit with Q1={self.Q1}, ε={self.epsilon}"
        for i in range(0, self.k):
            string += f"\nArm {i} --> N={self.N[i]}, Q={self.Q[i][-1]}, q={self.q[i]}"
        string += f"\nTotal reward = {sum(self.rewards)}"
        string += f"\nOptimal action count = {sum}"
        return string 
        
class GradientBandit(Bandit):
    def __init__(self, k, step_size, q_centre=0, use_baseline=False):
        super().__init__(k, step_size, q_centre)
        self.use_baseline = use_baseline
        self.reset()

    def reset(self):
        super().reset()
        self.H, self.probs = {}, {}
        self.reward_sum = 0
        for arm in range(0, self.k):
            self.H[arm] = [0]
            self.probs[arm] = []
         
    def run(self):
        for action, preference in self.H.items():
            self.probs[action].append(math.exp(preference[-1]) / (sum([math.exp(vals[-1]) for (key,vals) in self.H.items()])))
        action = random.choices(list(range(0, self.k)), [self.probs[arm][-1] for arm in range(0, self.k)], k=1)[0]
        self.N[action] += 1
        self.rewards.append(random.gauss(self.q[action], 1)) # reward is a normal distribution around the true reward
        self.reward_sum += self.rewards[-1]
        self.update(action)
        return self.rewards[-1]

    def update(self, action):
        baseline = self.reward_sum / len(self.rewards) if self.use_baseline else 0
        self.H[action].append(self.H[action][-1] + self.step_size * (self.rewards[-1] - baseline) * (1 - self.probs[action][-1]))
        for other_action in range(0, self.k):
            if other_action != action:
                self.H[other_action].append(self.H[other_action][-1] - self.step_size * (self.rewards[-1] - baseline) * self.probs[other_action][-1])
    def get_config(self):
        return f"k={self.k},a={self.step_size if self.step_size else '1/n'},baseline={self.use_baseline}"

    def __str__(self):
        string = f"{self.k}-armed gradient bandit with a={self.step_size}, basline={self.use_baseline}"
        for i in range(0, self.k):
            string += f"\nArm {i} --> N={self.N[i]}, H={self.H[i][-1]}, q={self.q[i]}, probability={self.probs[i][-1]}"
        string += f"\nTotal reward = {sum(self.rewards)}"
        string += f"\nOptimal action count = {sum}"
        return string 
