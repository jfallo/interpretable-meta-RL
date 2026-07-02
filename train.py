import torch
import torch.nn.functional as F
import numpy as np
import random
from DisRNN import MyDisRNN
from helper_functions import format_matrix, smooth
import matplotlib.pyplot as plt
import os

seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

os.makedirs(f'checkpoints/seed{seed}', exist_ok= True)
os.makedirs(f'figs/seed{seed}', exist_ok= True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define bandit task spaces
D = torch.rand
num_arms = 2


# initialize models and optimizers
input_size = 2

DisRNN_hidden_size = 5
DisRNN = MyDisRNN(DisRNN_hidden_size, input_size, num_arms).to(device)
DisRNN_critic = torch.nn.Linear(DisRNN_hidden_size, 1).to(device)
DisRNN_optimizer = torch.optim.Adam(
    list(DisRNN.parameters()) + list(DisRNN_critic.parameters()), 
    lr= 5e-3
)

LSTM_hidden_size = 48
LSTM = torch.nn.LSTM(input_size, LSTM_hidden_size).to(device)
LSTM_readout = torch.nn.Linear(LSTM_hidden_size, num_arms).to(device)
LSTM_critic = torch.nn.Linear(LSTM_hidden_size, 1).to(device)
LSTM_optimizer = torch.optim.Adam(
    list(LSTM.parameters()) + list(LSTM_readout.parameters()) + list(LSTM_critic.parameters()), 
    lr= 1e-3
)


# training hyperparameters
episodes = 50_000
batch_size = 32
batch_idx = torch.arange(batch_size, device= device)
T = 100
drift = 0.02
beta_e = 1.0
anneal_end = 2000
beta_v = 0.05
beta_max = 1e-3
warmup_end = 10_000


# training
DisRNN_regret_history = []
LSTM_regret_history = []
for ep in range(episodes):
    # sample task
    probs = D(batch_size, num_arms, device= device)


    # reset DisRNN state
    DisRNN.train()
    DisRNN_optimizer.zero_grad()

    DisRNN_h = torch.zeros(batch_size, DisRNN_hidden_size, device= device)
    DisRNN_x = torch.zeros(batch_size, input_size, device= device)

    DisRNN_log_probs = []
    DisRNN_rewards = []
    DisRNN_expected_rewards = []
    DisRNN_entropies = []
    DisRNN_bottleneck_losses = {'h': [], 'x': [], 'z': []}
    DisRNN_regrets = []

    # reset LSTM state
    LSTM.train()
    LSTM_optimizer.zero_grad()

    LSTM_h = torch.zeros(1, batch_size, LSTM_hidden_size, device= device)
    LSTM_c = torch.zeros(1, batch_size, LSTM_hidden_size, device= device)
    LSTM_x = torch.zeros(batch_size, input_size, device= device)

    LSTM_log_probs = []
    LSTM_rewards = []
    LSTM_expected_rewards = []
    LSTM_entropies = []
    LSTM_regrets = []


    for t in range(T):
        if t % 50 == 0:
            DisRNN_h = DisRNN_h.detach()
            LSTM_h = LSTM_h.detach()
            LSTM_c = LSTM_c.detach()


        # DisRNN step
        DisRNN_h, kls = DisRNN.step(DisRNN_h, DisRNN_x)
        DisRNN_logits = DisRNN.out(DisRNN_h)

        DisRNN_pi = torch.distributions.Categorical(logits= DisRNN_logits)
        DisRNN_a = DisRNN_pi.sample()
        DisRNN_r = (torch.rand(batch_size, device= device) < probs[batch_idx, DisRNN_a]).float()

        DisRNN_log_probs.append(DisRNN_pi.log_prob(DisRNN_a))
        DisRNN_rewards.append(DisRNN_r)
        DisRNN_expected_rewards.append(DisRNN_critic(DisRNN_h.detach()).squeeze(-1))
        DisRNN_entropies.append(DisRNN_pi.entropy())
        for key, val in kls.items():
            DisRNN_bottleneck_losses[key].append(val)
        DisRNN_regrets.append(probs.max(dim= -1).values - probs[batch_idx, DisRNN_a])

        DisRNN_x = torch.stack([2*DisRNN_a.float() - 1, 2*DisRNN_r - 1], dim= -1)


        # LSTM step
        LSTM_out, (LSTM_h, LSTM_c) = LSTM(LSTM_x.unsqueeze(0), (LSTM_h, LSTM_c))
        LSTM_logits = LSTM_readout(LSTM_out.squeeze(0))

        LSTM_pi = torch.distributions.Categorical(logits= LSTM_logits)
        LSTM_a = LSTM_pi.sample()
        LSTM_r = (torch.rand(batch_size, device= device) < probs[batch_idx, LSTM_a]).float()
        
        LSTM_log_probs.append(LSTM_pi.log_prob(LSTM_a))
        LSTM_rewards.append(LSTM_r)
        LSTM_expected_rewards.append(LSTM_critic(LSTM_out.squeeze(0).detach()).squeeze(-1))
        LSTM_entropies.append(LSTM_pi.entropy())
        LSTM_regrets.append(probs.max(dim= -1).values - probs[batch_idx, LSTM_a])

        LSTM_x = torch.stack([2*LSTM_a.float() - 1, 2*LSTM_r - 1], dim= -1)


        # drift
        probs += drift * torch.randn(batch_size, num_arms, device= device)
        probs = torch.clamp(probs, 0, 1)
        

    DisRNN_log_probs = torch.stack(DisRNN_log_probs)
    DisRNN_rewards = torch.stack(DisRNN_rewards)
    DisRNN_expected_rewards = torch.stack(DisRNN_expected_rewards)
    DisRNN_entropies = torch.stack(DisRNN_entropies)
    DisRNN_bottleneck_losses = {key: torch.stack(expected_rewards) for key, expected_rewards in DisRNN_bottleneck_losses.items()}
    DisRNN_regrets = torch.stack(DisRNN_regrets)

    LSTM_log_probs = torch.stack(LSTM_log_probs)
    LSTM_rewards = torch.stack(LSTM_rewards)
    LSTM_expected_rewards = torch.stack(LSTM_expected_rewards)
    LSTM_entropies = torch.stack(LSTM_entropies)
    LSTM_regrets = torch.stack(LSTM_regrets)


    # update betas
    beta_e = max(0.0, 1.0 - ep / anneal_end)
    beta = beta_max * min(ep / warmup_end, 1.0)
    

    # advantage actor-critic
    DisRNN_advantage = DisRNN_rewards - DisRNN_expected_rewards

    DisRNN_loss_policy = -(DisRNN_log_probs * DisRNN_advantage.detach()).mean()
    DisRNN_loss_entropy = DisRNN_entropies.mean()
    DisRNN_loss_critic = F.mse_loss(DisRNN_expected_rewards, DisRNN_rewards)
    DisRNN_loss_bottlenecks = sum(loss.mean() for loss in DisRNN_bottleneck_losses.values())
    DisRNN_loss = (
        DisRNN_loss_policy 
        - beta_e * DisRNN_loss_entropy 
        + beta_v * DisRNN_loss_critic
        + beta * DisRNN_loss_bottlenecks
    )
    
    DisRNN_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(DisRNN.parameters()) + list(DisRNN_critic.parameters()),
        max_norm= 1.0
    )
    DisRNN_optimizer.step()


    LSTM_advantage = LSTM_rewards - LSTM_expected_rewards

    LSTM_loss_policy = -(LSTM_log_probs * LSTM_advantage.detach()).mean()
    LSTM_loss_entropy = LSTM_entropies.mean()
    LSTM_loss_critic = F.mse_loss(LSTM_expected_rewards, LSTM_rewards)
    LSTM_loss = (
        LSTM_loss_policy
        - beta_e * LSTM_loss_entropy
        + beta_v * LSTM_loss_critic
    )

    LSTM_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(LSTM.parameters()) + list(LSTM_readout.parameters()) + list(LSTM_critic.parameters()),
        max_norm= 1.0
    )
    LSTM_optimizer.step()


    DisRNN_avg_R = DisRNN_rewards.mean().item()
    LSTM_avg_R = LSTM_rewards.mean().item()

    DisRNN_regret_history.append(DisRNN_regrets.mean().item())
    LSTM_regret_history.append(LSTM_regrets.mean().item())
    

    if ep % 250 == 0:
        print(f'ep {ep:5d}')
        print(f'  LSTM avg R {LSTM_avg_R:.3f} |   LSTM loss {LSTM_loss.item():.4f}')
        print(
            f'DisRNN avg R {DisRNN_avg_R:.3f} | DisRNN loss {DisRNN_loss.item():.4f} | '
            f'CE loss {DisRNN_loss_policy.item():.4f} | KL loss {DisRNN_loss_bottlenecks.item():.4f} | beta {beta:.2e} | '
            
        )
        M_h = torch.sigmoid(DisRNN.logit_M_h).detach().cpu().numpy()
        M_x = torch.sigmoid(DisRNN.logit_M_x).detach().cpu().numpy()
        M_z = torch.sigmoid(DisRNN.logit_M_z).detach().cpu().numpy()
        print()
        print(format_matrix(M_h, 'M_h', row_prefix= 'rule', col_prefix= 'lat'))
        print()
        print(format_matrix(M_x, 'M_x', row_prefix= 'rule', col_prefix= 'obs'))
        print()
        print(format_matrix(M_z.reshape(1,-1), 'M_z', row_prefix= 'lat', col_prefix= 'lat'))
        print()
        print()

    if (ep % 5000 == 0 and ep > 0) or ep == episodes - 1:
        torch.save({
            'ep': ep,
            'DisRNN_state_dict': DisRNN.state_dict(),
            'DisRNN_critic_state_dict': DisRNN_critic.state_dict(),
            'DisRNN_optimizer_state_dict': DisRNN_optimizer.state_dict(),
            'LSTM_state_dict': LSTM.state_dict(),
            'LSTM_readout_state_dict': LSTM_readout.state_dict(),
            'LSTM_critic_state_dict': LSTM_critic.state_dict(),
            'LSTM_optimizer_state_dict': LSTM_optimizer.state_dict(),
        }, f'checkpoints/seed{seed}/checkpoint_ep{ep}.pt')


        DisRNN_regret = np.array(DisRNN_regret_history)
        LSTM_regret = np.array(LSTM_regret_history)
        
        plt.figure(figsize= (8,5))
        plt.plot(smooth(DisRNN_regret), label= 'DisRNN', color= 'green')
        plt.plot(DisRNN_regret, alpha= 0.15, color= 'green')
        plt.plot(smooth(LSTM_regret), label= 'LSTM', color= 'black')
        plt.plot(LSTM_regret, alpha= 0.15, color= 'black')
        plt.xlabel('Episode')
        plt.ylabel('Regret')
        plt.title('Model Regret Over Time')
        plt.legend()
        plt.savefig(f'figs/seed{seed}/regret_ep{ep}.png')
        plt.close()
