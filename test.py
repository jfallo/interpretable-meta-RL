from config import *
from models import *

from SMPyBandits.Policies import Thompson, UCB
from SMPyBandits.Policies.Posterior import Beta
from Gittins import compute_gittins_table, Gittins


# load best models
best_DisRNN = torch.load(f'checkpoints/seed{seed}/best_DisRNN.pt')
DisRNN.load_state_dict(best_DisRNN['DisRNN_state_dict'])

best_LSTM = torch.load(f'checkpoints/seed{seed}/best_LSTM.pt')
LSTM.load_state_dict(best_LSTM['LSTM_state_dict'])
LSTM_readout.load_state_dict(best_LSTM['LSTM_readout_state_dict'])

# build gittins table
gittins_table = compute_gittins_table(max_total= trials+1, gamma= gamma, N= 200, tol= 1e-4)


# testing
DisRNN_cumulative_regrets = []
LSTM_cumulative_regrets = []
thompson_cumulative_regrets = []
ucb_cumulative_regrets = []
gittins_cumulative_regrets = []

num_tests = 300
for _ in range(num_tests):
    p = D(num_arms)
    probs = p.unsqueeze(0).to(device)
    
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
    gittins = Gittins(num_arms, gittins_table)


    DisRNN_regrets = []
    LSTM_regrets = []
    thompson_regrets = []
    ucb_regrets = []
    gittins_regrets = []
    with torch.no_grad():
        for t in range(trials):
            optimal = probs.max(dim= -1).values
            t_obs = torch.full((1, ), (t+1)/trials, device= device)

            # a single reward outcome for all agents for fair evaluation
            arm_rewards = torch.bernoulli(p).numpy()

            # DisRNN step
            DisRNN_h, kls = DisRNN.step(DisRNN_h, DisRNN_x)
            DisRNN_logits = DisRNN.out(DisRNN_h)

            DisRNN_pi = torch.distributions.Categorical(logits= DisRNN_logits)
            DisRNN_a = DisRNN_pi.sample()
            DisRNN_r = torch.tensor(arm_rewards[DisRNN_a.item()], device= device).unsqueeze(0)
            DisRNN_x = torch.stack([2*DisRNN_a.float() - 1, 2*DisRNN_r - 1, t_obs], dim= -1)
            DisRNN_regrets.append((optimal - probs[0, DisRNN_a]).cpu())

            # LSTM step
            LSTM_out, (LSTM_h, LSTM_c) = LSTM(LSTM_x.unsqueeze(0), (LSTM_h, LSTM_c))
            LSTM_logits = LSTM_readout(LSTM_out.squeeze(0))

            LSTM_pi = torch.distributions.Categorical(logits= LSTM_logits)
            LSTM_a = LSTM_pi.sample()
            LSTM_r = torch.tensor(arm_rewards[LSTM_a.item()], device= device).unsqueeze(0)
            LSTM_x = torch.stack([2*LSTM_a.float() - 1, 2*LSTM_r - 1, t_obs], dim= -1)
            LSTM_regrets.append((optimal - probs[0, LSTM_a]).cpu())

            # Thompson step
            thompson_a = thompson.choice()
            thompson_r = arm_rewards[thompson_a]
            thompson.getReward(thompson_a, thompson_r)
            thompson_regrets.append(optimal.item() - p[thompson_a].item())

            # UCB step
            ucb_a = ucb.choice()
            ucb_r = arm_rewards[ucb_a]
            ucb.getReward(ucb_a, ucb_r)
            ucb_regrets.append(optimal.item() - p[ucb_a].item())

            # Gittins step
            gittins_a = gittins.choice()
            gittins_r = arm_rewards[gittins_a]
            gittins.getReward(gittins_a, gittins_r)
            gittins_regrets.append(optimal.item() - p[gittins_a].item())
            

    DisRNN_cumulative_regrets.append(np.array(DisRNN_regrets).cumsum())
    LSTM_cumulative_regrets.append(np.array(LSTM_regrets).cumsum())
    thompson_cumulative_regrets.append(np.array(thompson_regrets).cumsum())
    ucb_cumulative_regrets.append(np.array(ucb_regrets).cumsum())
    gittins_cumulative_regrets.append(np.array(gittins_regrets).cumsum())




def plot_agent(data, color, linestyle, label):
    mean = np.stack(data).mean(axis= 0)
    std = np.stack(data).std(axis= 0)
    plt.plot(mean, color= color, linestyle= linestyle, label= label)
    #plt.fill_between(range(trials), mean - std, mean + std, alpha= 0.1, color= color, linestyle= linestyle)

plt.figure(figsize= (8,5))
plot_agent(DisRNN_cumulative_regrets, 'blue', '-', 'DisRNN')
plot_agent(LSTM_cumulative_regrets, 'green', '-', 'LSTM')
plot_agent(thompson_cumulative_regrets, 'gray', '--', 'Thompson')
plot_agent(ucb_cumulative_regrets, 'lightgray', '--', 'UCB')
plot_agent(gittins_cumulative_regrets, 'black', '--', 'Gittins')
plt.xlabel('Trial')
plt.ylabel('Cumulative Regret')
plt.title('Model Cumulative Regret')
plt.legend()
plt.grid()
plt.savefig(f'figs/seed{seed}/cumulative_regret.png')
plt.close()
