from config import *
from helpers import print_bottleneck_parameters


def latents_analysis(config, checkpoints_path, figs_path):
    # set task
    D = config['D']
    num_trials = config['num_trials']
    restless = config['restless']
    drift = config['drift']
    dependent_arms = config['dependent_arms']


    # initialize DisRNN
    input_size = config['input_size']
    DisRNN_hidden_size = config['hidden_size']['DisRNN']
    DisRNN = MyDisRNN(DisRNN_hidden_size, input_size, num_arms).to(device)

    # load best DisRNN
    best_DisRNN = torch.load(checkpoints_path + 'best_DisRNN.pt')
    DisRNN.load_state_dict(best_DisRNN['DisRNN_state_dict'])
    print_bottleneck_parameters(DisRNN)


    # latents analysis
    ex_sessions = 3
    for session in range(ex_sessions):
        # set task
        probs = D(1, num_arms, device)
                
        # reset DisRNN state
        DisRNN.eval()
        h = torch.zeros(1, DisRNN_hidden_size, device= device)
        x = torch.zeros(1, input_size, device= device)

        latent_history = []
        action_history = []
        reward_history = []
        with torch.no_grad():
            for t in range(num_trials):
                arm_rewards = torch.bernoulli(probs).squeeze(0)

                # DisRNN step
                h, _ = DisRNN.step(h, x)
                logits = DisRNN.out(h)

                pi = torch.distributions.Categorical(logits= logits)
                a = pi.sample()
                r = arm_rewards[a.item()].unsqueeze(0)
                x = torch.stack([2*a.float() - 1, 2*r - 1], dim= -1)

                # track latents
                latent_history.append(h)
                action_history.append(a)
                reward_history.append(r)


                # restless bandits
                if restless:
                    probs += drift * torch.randn(1, num_arms, device= device)
                    probs = torch.clamp(probs, 0, 1)
                    if dependent_arms:
                        probs[:, 1] = 1 - probs[:, 0]


        latent_history = torch.stack(latent_history).squeeze(1).cpu().numpy()
        action_history = torch.stack(action_history).cpu().numpy()
        reward_history = torch.stack(reward_history).cpu().numpy()


        # plot latent trajectories with (a,r) markers
        plt.figure(figsize= (12,6))
        for h in range(DisRNN_hidden_size):
            plt.plot(latent_history[:, h], label= f'Latent {h}')
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        margin = 0.02 * (ymax - ymin)
        y_top = ymax + margin
        y_bottom = ymin - margin
        for t in range(num_trials):
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
        plt.savefig(figs_path + f'trajectories_ex{session+1}.png')
        plt.close()


    # plot latent updates
    conditions = [
        ('Left, Unrewarded',  -1, -1),
        ('Left, Rewarded',    -1,  1),
        ('Right, Unrewarded',  1, -1),
        ('Right, Rewarded',    1,  1),
    ]
    active_rules = [
        rule for rule in range(DisRNN_hidden_size)
        if (torch.sigmoid(DisRNN.logit_M_x[rule]) > 0.1).any()
    ]

    low, high = -2.0, 2.0
    h_prevs = torch.linspace(low, high, 100, device= device)

    times = [50]
    for t in times:
        fig, axes = plt.subplots(len(active_rules), 4, figsize= (12, 3*len(active_rules)), sharex= True, sharey= True)
        if len(active_rules) == 1:
            axes = axes[np.newaxis, :]

        M_h = torch.sigmoid(DisRNN.logit_M_h)
        M_x = torch.sigmoid(DisRNN.logit_M_x)

        with torch.no_grad():
            for i, rule in enumerate(active_rules):
                for col, (label, a, r) in enumerate(conditions):
                    ax = axes[i, col]

                    h = torch.zeros(len(h_prevs), DisRNN_hidden_size, device= device)
                    h[:, rule] = h_prevs
                    x = torch.tensor([[a, r]], dtype= torch.float32, device= device).repeat(len(h_prevs), 1)

                    h = M_h[rule].unsqueeze(0) * h  # no noise
                    x = M_x[rule].unsqueeze(0) * x

                    z = torch.cat([h, x], dim= -1)
                    logit_w, u = DisRNN.updateMLPs[rule](z).unbind(dim= -1)
                    w = torch.sigmoid(logit_w)

                    z_out = (1 - w) * h_prevs + w * u

                    ax.plot([low, high], [low, high], 'k--', linewidth= 1)
                    ax.set_xlabel(f'Latent {rule}')
                    ax.axhline(0, color= 'k', linewidth= 0.8)
                    ax.axvline(0, color= 'k', linewidth= 0.8)
                    ax.plot(h_prevs.cpu(), z_out.cpu(), linewidth= 2.5, color= f'C{rule}')
                    if i == 0:
                        ax.set_title(label)
                    if col == 0:
                        ax.set_ylabel(f'Updated Latent {rule}')

        fig.suptitle(f'Latent Updates at Trial {t}', fontsize= 14)
        plt.tight_layout()
        plt.savefig(figs_path + f'updates_at_trial{t}.png')
        plt.close()




def main():
    for exp, config in exps.items():
        checkpoints_path = f'checkpoints/{exp}/seed{seed}/'
        figs_path = f'figs/{exp}/seed{seed}/latents/'

        if os.path.exists(checkpoints_path):
            analysis_res = input(f"Begin latents analysis for experiment: {exp}, seed {seed}? (y/n):")
            if analysis_res.lower() == 'n':
                continue

            if os.path.exists(figs_path):
                overwrite_res = input(f"There is history for this experiment. Do you want to overwrite it? (y/n): ")
                if overwrite_res.lower() == 'n':
                    continue

            print(f"Beginning latents analysis for experiment {exp} bandits, seed {seed}.\n")
            latents_analysis(config, checkpoints_path, figs_path)


if __name__ == "__main__":
    main()
    