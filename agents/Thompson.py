import numpy as np


class Thompson:
    def __init__(self, num_arms):
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)
    
    def choice(self):
        samples = np.random.beta(self.alpha, self.beta)
        
        return int(np.argmax(samples))
    
    def getReward(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
            