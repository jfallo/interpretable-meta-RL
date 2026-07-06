from config import *
from models import *

import matplotlib.pyplot as plt

from SMPyBandits.Arms import Bernoulli
from SMPyBandits.Environment import MAB
from SMPyBandits.Policies import Thompson, UCB
from SMPyBandits.Policies.Posterior import Beta


# load trained models
checkpoint = torch.load(f'checkpoints/seed{seed}/checkpoint_ep{episodes - 1}.pt')
DisRNN.load_state_dict(checkpoint['DisRNN_state_dict'])
LSTM.load_state_dict(checkpoint['LSTM_state_dict'])
LSTM_readout.load_state_dict(checkpoint['LSTM_readout_state_dict'])


# testing
num_tests = 150

DisRNN_cumulative_regrets = []
LSTM_cumulative_regrets = []
Thompson_cumulative_regrets = []
UCB_cumulative_regrets = []
for _ in range(num_tests):
    p = D(num_arms)
    probs = p.unsqueeze(0).to(device)
    env = MAB([Bernoulli(p[i].item()) for i in range(num_arms)])
    

    # reset DisRNN state
    DisRNN.eval()
    DisRNN_h = torch.zeros(1, DisRNN_hidden_size, device= device)
    DisRNN_x = torch.zeros(1, input_size, device= device)

    # reset LSTM state
    LSTM.eval()
    LSTM_h = torch.zeros(1, 1, LSTM_hidden_size, device= device)
    LSTM_c = torch.zeros(1, 1, LSTM_hidden_size, device= device)
    LSTM_x = torch.zeros(1, input_size, device= device)

    # test models
    thompson = Thompson(num_arms, Beta)
    ucb = UCB(num_arms)


    DisRNN_regrets = []
    LSTM_regrets = []
    thompson_regrets = []
    ucb_regrets = []
    with torch.no_grad():
        for t in range(T):
            optimal = probs.max(dim= -1).values

            # DisRNN step
            DisRNN_h, kls = DisRNN.step(DisRNN_h, DisRNN_x)
            DisRNN_logits = DisRNN.out(DisRNN_h)

            DisRNN_pi = torch.distributions.Categorical(logits= DisRNN_logits)
            DisRNN_a = DisRNN_pi.sample()
            DisRNN_r = (torch.rand(1, device= device) < probs[0, DisRNN_a]).float()
            DisRNN_x = torch.stack([2*DisRNN_a.float() - 1, 2*DisRNN_r - 1], dim= -1)
            DisRNN_regrets.append((optimal - probs[0, DisRNN_a]).cpu())

            # LSTM step
            LSTM_out, (LSTM_h, LSTM_c) = LSTM(LSTM_x.unsqueeze(0), (LSTM_h, LSTM_c))
            LSTM_logits = LSTM_readout(LSTM_out.squeeze(0))

            LSTM_pi = torch.distributions.Categorical(logits= LSTM_logits)
            LSTM_a = LSTM_pi.sample()
            LSTM_r = (torch.rand(1, device= device) < probs[0, LSTM_a]).float()
            LSTM_x = torch.stack([2*LSTM_a.float() - 1, 2*LSTM_r - 1], dim= -1)
            LSTM_regrets.append((optimal - probs[0, LSTM_a]).cpu())

            # Thompson step
            thompson_a = thompson.choice()
            thompson_r = env.draw(thompson_a)
            thompson.getReward(thompson_a, thompson_r)
            thompson_regrets.append(optimal.item() - p[thompson_a].item())

            # UCB step
            ucb_a = ucb.choice()
            ucb_r = env.draw(ucb_a)
            ucb.getReward(ucb_a, ucb_r)
            ucb_regrets.append(optimal.item() - p[ucb_a].item())
            
            
    DisRNN_cumulative_regrets.append(np.array(DisRNN_regrets).cumsum())
    LSTM_cumulative_regrets.append(np.array(LSTM_regrets).cumsum())
    Thompson_cumulative_regrets.append(np.array(thompson_regrets).cumsum())
    UCB_cumulative_regrets.append(np.array(ucb_regrets).cumsum())




def plot_agent(data, color, label):
    mean = np.stack(data).mean(axis= 0)
    std = np.stack(data).std(axis= 0)
    plt.plot(mean, color= color, label= label)
    plt.fill_between(range(T), mean - std, mean + std, alpha= 0.1, color= color)

plt.figure(figsize= (8,5))
plot_agent(DisRNN_cumulative_regrets, 'green', 'DisRNN')
plot_agent(LSTM_cumulative_regrets, 'black', 'LSTM')
plot_agent(Thompson_cumulative_regrets, 'gray', 'Thompson')
plot_agent(UCB_cumulative_regrets, 'blue', 'UCB')
plt.xlabel('Trial')
plt.ylabel('Cumulative Regret')
plt.title('Model Cumulative Regret')
plt.legend()
plt.grid()
plt.savefig(f'figs/seed{seed}/cumulative_regret.png')
plt.close()
