from config import *


# load best DisRNN
best_DisRNN = torch.load(f'checkpoints/seed{seed}_test/best_DisRNN.pt')
DisRNN.load_state_dict(best_DisRNN['DisRNN_state_dict'])

# set task
probs = D(1, num_arms, device= device)
        
# reset DisRNN state
DisRNN.eval()
DisRNN_h = torch.zeros(1, DisRNN_hidden_size, device= device)
DisRNN_x = torch.zeros(1, input_size, device= device)

latent_history = []
action_history = []
reward_history = []
with torch.no_grad():
    for t in range(trials):
        arm_rewards = torch.bernoulli(probs).squeeze(0)

        # DisRNN step
        t_obs = torch.full((1, ), (t+1)/trials, device= device)

        DisRNN_h, _ = DisRNN.step(DisRNN_h, DisRNN_x)
        DisRNN_logits = DisRNN.out(DisRNN_h)

        DisRNN_pi = torch.distributions.Categorical(logits= DisRNN_logits)
        DisRNN_a = DisRNN_pi.sample()
        DisRNN_r = arm_rewards[DisRNN_a.item()].unsqueeze(0)
        DisRNN_x = torch.stack([2*DisRNN_a.float() - 1, 2*DisRNN_r - 1, t_obs], dim= -1)

        # track latents
        latent_history.append(DisRNN_h)
        action_history.append(DisRNN_a)
        reward_history.append(DisRNN_r)

latent_history = torch.stack(latent_history).squeeze(1).cpu().numpy()
action_history = torch.stack(action_history).cpu().numpy()
reward_history = torch.stack(reward_history).cpu().numpy()

# plot latent trajectories
plt.figure(figsize= (12,6))
for h in range(DisRNN_hidden_size):
    plt.plot(latent_history[:, h], label= f'Latent {h}')

# plot (action, reward) markers
ax = plt.gca()
ymin, ymax = ax.get_ylim()
margin = 0.02 * (ymax - ymin)
y_top = ymax + margin
y_bottom = ymin - margin
for t in range(trials):
    pos = y_top if action_history[t] == 0 else y_bottom
    color = 'green' if reward_history[t] == 1 else 'red'
    plt.plot(t, pos, marker= '|', markersize= 8, markeredgewidth= 4, color= color, linestyle= 'None', zorder= 10)

plt.ylim(y_bottom - margin, y_top + margin)
plt.xlabel('Trial')
plt.title('Example Session')
plt.text(-8, y_top, 'Left Choices', ha= 'right', va= 'center')
plt.text(-8, y_bottom, 'Right Choices', ha= 'right', va= 'center')
plt.legend(loc= 'center left', bbox_to_anchor= (1.02, 0.5), borderaxespad= 0.0)
plt.tight_layout(rect= [0, 0, 0.85, 1])
plt.show()
plt.show()
