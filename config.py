import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from agents.DisRNN import MyDisRNN


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = torch.rand
num_arms = 2
trials = 100
gamma = 0.95


input_size = 3

DisRNN_hidden_size = 5
DisRNN = MyDisRNN(DisRNN_hidden_size, input_size, num_arms).to(device)

LSTM_hidden_size = 48
LSTM = torch.nn.LSTM(input_size, LSTM_hidden_size).to(device)
LSTM_readout = torch.nn.Linear(LSTM_hidden_size, num_arms).to(device)
