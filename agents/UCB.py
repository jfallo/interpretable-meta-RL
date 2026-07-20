import numpy as np


class UCB:
    def __init__(self, num_arms, c):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms, dtype= int)
        self.rewards = np.zeros(num_arms)
        self.c = c
        self.total = 0
    
    def choice(self):
        self.total += 1
        
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return int(untried[0])
        
        means = self.rewards / self.counts
        bounds = np.sqrt(self.c * np.log(self.total) / self.counts)
        
        return int(np.argmax(means + bounds))
    
    def getReward(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        