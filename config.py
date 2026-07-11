import numpy as np
import torch
import random
import matplotlib.pyplot as plt

seed = 46
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = torch.rand
num_arms = 2
trials = 100
