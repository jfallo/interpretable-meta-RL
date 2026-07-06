from config import *
from DisRNN import MyDisRNN

input_size = 2

DisRNN_hidden_size = 16
DisRNN = MyDisRNN(DisRNN_hidden_size, input_size, num_arms).to(device)

LSTM_hidden_size = 48
LSTM = torch.nn.LSTM(input_size, LSTM_hidden_size).to(device)
LSTM_readout = torch.nn.Linear(LSTM_hidden_size, num_arms).to(device)
