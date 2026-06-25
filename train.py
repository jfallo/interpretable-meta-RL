import torch
from DisRNN import DisRNN
from helper_functions import format_matrix
import matplotlib.pyplot as plt
import os
os.makedirs('checkpoints', exist_ok= True)


# set task space
D = torch.rand
num_arms = 2

# initialize DisRNN model and Adam optimizer
hidden_size = 5
input_size = 2
model = DisRNN(m= hidden_size, n= 2, q= num_arms)
optimizer = torch.optim.Adam(model.parameters(), lr= 5e-3)

# setup GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# training hyperparameters
episodes = 20_000
batch_size = 32
batch_idx = torch.arange(batch_size, device= device)
T = 500
drift = 0.02
beta_max = 1e-3
beta_warmup_start = 1000
beta_warmup_end = 2000

for ep in range(episodes):
    model.train()
    optimizer.zero_grad()

    probs = D(batch_size, num_arms, device= device)

    h = torch.zeros(batch_size, hidden_size, device= device)
    x = torch.zeros(batch_size, input_size, device= device)

    log_probs = []
    rewards = []
    bottleneck_losses = {'h': [], 'x': [], 'z': []}

    for t in range(T):
        if t % 50 == 0:
            h = h.detach()

        h, kls = model.step(h, x)
        logits = model.out(h)

        # sample action and get rewards
        pi = torch.distributions.Categorical(logits= logits)
        a = pi.sample()
        log_prob = pi.log_prob(a)
        r = (torch.rand(batch_size, device= device) < probs[batch_idx, a]).float()
        
        # prepare next input
        x = torch.stack([2*a.float() - 1, 2*r - 1], dim= -1)

        # apply bounded random drift to probs
        probs += drift * torch.randn(batch_size, num_arms, device= device)
        probs = torch.clamp(probs, 0, 1)
        
        log_probs.append(log_prob)
        rewards.append(r)
        for key, val in kls.items():
            bottleneck_losses[key].append(val)
        
    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    bottleneck_losses = {key: torch.stack(vals) for key, vals in bottleneck_losses.items()}
    
    # REINFORCE with baseline
    baseline = rewards.mean(dim= 0, keepdim= True)
    advantage = rewards - baseline

    # update beta
    beta = beta_max * min((ep - beta_warmup_start) / (beta_warmup_end - beta_warmup_start), 1.0)
    beta = 0 if ep < beta_warmup_start else beta
    
    loss_rl = -(log_probs * advantage).mean()
    loss_bottleneck = sum(loss.mean() for loss in bottleneck_losses.values())
    loss = loss_rl + beta * loss_bottleneck
    
    loss.backward()
    optimizer.step()

    if ep % 250 == 0:
        mean_reward = rewards.mean().item()
    
        M_h = torch.sigmoid(model.logit_M_h).detach().cpu().numpy()
        M_x = torch.sigmoid(model.logit_M_x).detach().cpu().numpy()
        M_z = torch.sigmoid(model.logit_M_z).detach().cpu().numpy()

        print(
            f'ep {ep:5d} | loss {loss.item():.4f} | '
            f'RL {loss_rl.item():.4f} | KL {loss_bottleneck.item():.4f} | '
            f'beta {beta:.2e} | mean_r {mean_reward:.3f}'
        )
        print(format_matrix(M_h, 'M_h', row_prefix='rule', col_prefix='lat'))
        print(format_matrix(M_x, 'M_x', row_prefix='rule', col_prefix='obs'))
        print(format_matrix(M_z.reshape(1,-1), 'M_z', row_prefix='lat', col_prefix='lat'))
        print()

    if (ep % 1000 == 0 and ep > 0) or ep == episodes - 1:
        torch.save({
            'ep': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f'checkpoints/checkpoint_ep{ep}.pt')
