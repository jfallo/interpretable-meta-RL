from config import *
from helpers import print_bottleneck_parameters, smooth
import os


def train(config, checkpoints_path, figs_path):
    # set task
    D = config['D']
    num_trials = config['num_trials']


    # initialize training models and optimizers
    input_size = config['input_size']

    DisRNN_hidden_size = config['hidden_size']['DisRNN']
    DisRNN = MyDisRNN(DisRNN_hidden_size, input_size, num_arms).to(device)
    DisRNN_gamma = config['gamma']['DisRNN']
    DisRNN_lr = config['lr']['DisRNN']
    DisRNN_critic = torch.nn.Linear(DisRNN_hidden_size, 1).to(device)
    DisRNN_optimizer = torch.optim.Adam(
        list(DisRNN.parameters()) + list(DisRNN_critic.parameters()), 
        lr= DisRNN_lr
    )

    LSTM_hidden_size = config['hidden_size']['LSTM']
    LSTM = torch.nn.LSTM(input_size, LSTM_hidden_size).to(device)
    LSTM_readout = torch.nn.Linear(LSTM_hidden_size, num_arms).to(device)
    LSTM_gamma = config['gamma']['LSTM']
    LSTM_lr = config['lr']['LSTM']
    LSTM_critic = torch.nn.Linear(LSTM_hidden_size, 1).to(device)
    LSTM_optimizer = torch.optim.Adam(
        list(LSTM.parameters()) + list(LSTM_readout.parameters()) + list(LSTM_critic.parameters()), 
        lr= LSTM_lr
    )

    # training hyperparameters
    batch_size = config['batch_size']
    batch_idx = torch.arange(batch_size, device= device)
    steps_unrolled = config['steps_unrolled']
    beta_e_annealed = config['beta_e_annealed']
    if beta_e_annealed:
        beta_e_floor = config['beta_e']
        anneal_end = 5000
    else:
        beta_e = config['beta_e']
    beta_v = config['beta_v']
    beta_floor = config['beta_floor']
    beta_ceil = config['beta_ceil']
    warmup_start = 5000
    warmup_end = 10_000

    train_LSTM_until_ep = config['train_LSTM_until_ep']

    m_min = torch.logit(torch.tensor(0.01)).item()
    sigma_min = torch.log(torch.tensor(0.01)).item()


    # training helpers
    def plot_regret_history(DisRNN_history, LSTM_history, plot_name):
        plt.figure(figsize= (8,5))
        plt.plot(DisRNN_history, label= 'DisRNN', color= 'blue')
        plt.plot(LSTM_history, label= 'LSTM', color= 'green')
        plt.xlabel('Episode')
        plt.ylabel('Regret')
        plt.title('Model Regret Over Time')
        plt.legend()
        plt.grid()
        plt.savefig(figs_path + f'{plot_name}.png')
        plt.close()


    def disentangled(model, low= 0.1, high= 0.9):
        M_h = torch.sigmoid(model.logit_M_h).detach()
        M_x = torch.sigmoid(model.logit_M_x).detach()
        M_z = torch.sigmoid(model.logit_M_z).detach()
        
        h_converged = ((M_h <= low) | (M_h >= high)).all()
        x_converged = ((M_x <= low) | (M_x >= high)).all()
        z_converged = ((M_z <= low) | (M_z >= high)).all()
        
        return h_converged and x_converged and z_converged


    def run_training_episode(phase, train_LSTM):
        # sample task
        probs = D(batch_size, num_arms, device)

        # reset DisRNN state
        DisRNN.train()
        DisRNN_optimizer.zero_grad()

        DisRNN_h = torch.zeros(batch_size, DisRNN_hidden_size, device= device)
        DisRNN_x = torch.zeros(batch_size, input_size, device= device)

        DisRNN_log_probs = []
        DisRNN_rewards = []
        DisRNN_expected_returns = []
        DisRNN_entropies = []
        DisRNN_bottleneck_losses = {'h': [], 'x': [], 'z': []}
        DisRNN_regrets = []

        # reset LSTM state
        if train_LSTM:
            LSTM.train()
            LSTM_optimizer.zero_grad()

            LSTM_h = torch.zeros(1, batch_size, LSTM_hidden_size, device= device)
            LSTM_c = torch.zeros(1, batch_size, LSTM_hidden_size, device= device)
            LSTM_x = torch.zeros(batch_size, input_size, device= device)

            LSTM_log_probs = []
            LSTM_rewards = []
            LSTM_expected_returns = []
            LSTM_entropies = []
            LSTM_regrets = []


        for t in range(num_trials):
            if t % steps_unrolled == 0:
                DisRNN_h = DisRNN_h.detach()
                if train_LSTM:
                    LSTM_h = LSTM_h.detach()
                    LSTM_c = LSTM_c.detach()


            # DisRNN step
            DisRNN_h, kls = DisRNN.step(DisRNN_h, DisRNN_x)
            DisRNN_logits = DisRNN.out(DisRNN_h)

            DisRNN_pi = torch.distributions.Categorical(logits= DisRNN_logits)
            DisRNN_a = DisRNN_pi.sample()
            DisRNN_r = (torch.rand(batch_size, device= device) < probs[batch_idx, DisRNN_a]).float()
            DisRNN_x = torch.stack([2*DisRNN_a.float() - 1, 2*DisRNN_r - 1], dim= -1)

            DisRNN_log_probs.append(DisRNN_pi.log_prob(DisRNN_a))
            DisRNN_rewards.append(DisRNN_r)
            DisRNN_expected_returns.append(DisRNN_critic(DisRNN_h.detach()).squeeze(-1))
            DisRNN_entropies.append(DisRNN_pi.entropy())
            for key, val in kls.items():
                DisRNN_bottleneck_losses[key].append(val)
            DisRNN_regrets.append(probs.max(dim= -1).values - probs[batch_idx, DisRNN_a])

            # LSTM step
            if train_LSTM:
                LSTM_out, (LSTM_h, LSTM_c) = LSTM(LSTM_x.unsqueeze(0), (LSTM_h, LSTM_c))
                LSTM_logits = LSTM_readout(LSTM_out.squeeze(0))

                LSTM_pi = torch.distributions.Categorical(logits= LSTM_logits)
                LSTM_a = LSTM_pi.sample()
                LSTM_r = (torch.rand(batch_size, device= device) < probs[batch_idx, LSTM_a]).float()
                LSTM_x = torch.stack([2*LSTM_a.float() - 1, 2*LSTM_r - 1], dim= -1)
                
                LSTM_log_probs.append(LSTM_pi.log_prob(LSTM_a))
                LSTM_rewards.append(LSTM_r)
                LSTM_expected_returns.append(LSTM_critic(LSTM_out.squeeze(0)).squeeze(-1))
                LSTM_entropies.append(LSTM_pi.entropy())
                LSTM_regrets.append(probs.max(dim= -1).values - probs[batch_idx, LSTM_a])
            

        DisRNN_log_probs = torch.stack(DisRNN_log_probs)
        DisRNN_rewards = torch.stack(DisRNN_rewards)
        DisRNN_expected_returns = torch.stack(DisRNN_expected_returns)
        DisRNN_entropies = torch.stack(DisRNN_entropies)
        DisRNN_bottleneck_losses = {key: torch.stack(vals) for key, vals in DisRNN_bottleneck_losses.items()}
        DisRNN_regrets = torch.stack(DisRNN_regrets)
        
        DisRNN_regret = DisRNN_regrets.mean().item()
        DisRNN_total_reward = DisRNN_rewards.sum(dim= 0).mean().item()

        if train_LSTM:
            LSTM_log_probs = torch.stack(LSTM_log_probs)
            LSTM_rewards = torch.stack(LSTM_rewards)
            LSTM_expected_returns = torch.stack(LSTM_expected_returns)
            LSTM_entropies = torch.stack(LSTM_entropies)
            LSTM_regrets = torch.stack(LSTM_regrets)

            LSTM_regret = LSTM_regrets.mean().item()
            LSTM_total_reward = LSTM_rewards.sum(dim= 0).mean().item()
        else:
            LSTM_regret = float('NaN')
            LSTM_total_reward = float('NaN')

        
        # --- advantage actor-critic ------

        # update betas
        if beta_e_annealed:
            beta_e = beta_e_floor + (1.0 - beta_e_floor) * max(0.0, 1.0 - ep / anneal_end)
        if ep < warmup_start or phase == 2:
            beta = beta_floor
        else:
            beta = beta_floor + (beta_ceil - beta_floor) * min((ep - warmup_start) / (warmup_end - warmup_start), 1.0)

        # DisRNN update
        DisRNN_returns = DisRNN_rewards.clone()
        for t in reversed(range(num_trials - 1)):
            DisRNN_returns[t] = DisRNN_rewards[t] + DisRNN_gamma * DisRNN_returns[t+1]
        DisRNN_returns = (DisRNN_returns - DisRNN_returns.mean(dim= 1, keepdim= True)) / (DisRNN_returns.std(dim= 1, keepdim= True) + 1e-8)
        DisRNN_advantage = DisRNN_returns - DisRNN_expected_returns
        
        DisRNN_loss_actor = -(DisRNN_log_probs * DisRNN_advantage.detach()).mean()
        DisRNN_loss_critic = torch.nn.functional.mse_loss(DisRNN_expected_returns, DisRNN_returns)
        DisRNN_loss_entropy = DisRNN_entropies.mean()
        DisRNN_loss_bottlenecks = sum(loss.mean() for loss in DisRNN_bottleneck_losses.values())
        DisRNN_loss = (
            DisRNN_loss_actor 
            + beta_v * DisRNN_loss_critic
            - beta_e * DisRNN_loss_entropy 
            + beta * DisRNN_loss_bottlenecks
        )
        
        DisRNN_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(DisRNN.parameters()) + list(DisRNN_critic.parameters()),
            max_norm= 1.0
        )
        DisRNN_optimizer.step()
        with torch.no_grad():
            DisRNN.logit_M_h.clamp_(min= m_min)
            DisRNN.log_sigma_h.clamp_(min= sigma_min, max= 0.0)
            DisRNN.logit_M_x.clamp_(min= m_min)
            DisRNN.log_sigma_x.clamp_(min= sigma_min, max= 0.0)
            DisRNN.logit_M_z.clamp_(min= m_min)
            DisRNN.log_sigma_z.clamp_(min= sigma_min, max= 0.0)

        # LSTM update
        if train_LSTM:
            LSTM_returns = LSTM_rewards.clone()
            for t in reversed(range(num_trials - 1)):
                LSTM_returns[t] = LSTM_rewards[t] + LSTM_gamma * LSTM_returns[t+1]
            LSTM_returns = (LSTM_returns - LSTM_returns.mean(dim= 1, keepdim= True)) / (LSTM_returns.std(dim= 1, keepdim= True) + 1e-8)
            LSTM_advantage = LSTM_returns - LSTM_expected_returns

            LSTM_loss_actor = -(LSTM_log_probs * LSTM_advantage.detach()).mean()
            LSTM_loss_critic = torch.nn.functional.mse_loss(LSTM_expected_returns, LSTM_returns)
            LSTM_loss_entropy = LSTM_entropies.mean()
            LSTM_loss = (
                LSTM_loss_actor
                + beta_v * LSTM_loss_critic
                - beta_e * LSTM_loss_entropy
            )

            LSTM_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(LSTM.parameters()) + list(LSTM_readout.parameters()) + list(LSTM_critic.parameters()),
                max_norm= 1.0
            )
            LSTM_optimizer.step()


        return DisRNN_regret, DisRNN_total_reward, LSTM_regret, LSTM_total_reward


    def run_eval_episode():
        with torch.no_grad():
            # sample task
            probs = D(batch_size, num_arms, device)

            # reset DisRNN state
            DisRNN.eval()

            DisRNN_h = torch.zeros(batch_size, DisRNN_hidden_size, device= device)
            DisRNN_x = torch.zeros(batch_size, input_size, device= device)

            DisRNN_eval_ep_regrets = []

            # reset LSTM state
            LSTM.eval()
            
            LSTM_h = torch.zeros(1, batch_size, LSTM_hidden_size, device= device)
            LSTM_c = torch.zeros(1, batch_size, LSTM_hidden_size, device= device)
            LSTM_x = torch.zeros(batch_size, input_size, device= device)

            LSTM_eval_ep_regrets = []
            
            for t in range(num_trials):
                optimal = probs.max(dim= -1).values

                # DisRNN step
                DisRNN_h, _ = DisRNN.step(DisRNN_h, DisRNN_x)
                DisRNN_logits = DisRNN.out(DisRNN_h)

                DisRNN_pi = torch.distributions.Categorical(logits= DisRNN_logits)
                DisRNN_a = DisRNN_pi.sample()
                DisRNN_r = (torch.rand(batch_size, device= device) < probs[batch_idx, DisRNN_a]).float()
                DisRNN_x = torch.stack([2*DisRNN_a.float() - 1, 2*DisRNN_r - 1], dim= -1)
                DisRNN_eval_ep_regrets.append((optimal - probs[batch_idx, DisRNN_a]).cpu())

                # LSTM step
                LSTM_out, (LSTM_h, LSTM_c) = LSTM(LSTM_x.unsqueeze(0), (LSTM_h, LSTM_c))
                LSTM_logits = LSTM_readout(LSTM_out.squeeze(0))

                LSTM_pi = torch.distributions.Categorical(logits= LSTM_logits)
                LSTM_a = LSTM_pi.sample()
                LSTM_r = (torch.rand(batch_size, device= device) < probs[batch_idx, LSTM_a]).float()
                LSTM_x = torch.stack([2*LSTM_a.float() - 1, 2*LSTM_r - 1], dim= -1)
                LSTM_eval_ep_regrets.append((optimal - probs[batch_idx, LSTM_a]).cpu())

        return np.mean(DisRNN_eval_ep_regrets), np.mean(LSTM_eval_ep_regrets)




    # --- Phase 1: train until disentanglement ------
    DisRNN_regret_history = []
    LSTM_regret_history = []

    ep = 0
    while not disentangled(DisRNN):
        train_LSTM = ep < train_LSTM_until_ep
        DisRNN_regret, DisRNN_total_reward, LSTM_regret, LSTM_total_reward = run_training_episode(phase= 1, train_LSTM= train_LSTM)
        DisRNN_regret_history.append(DisRNN_regret)
        if train_LSTM:
            LSTM_regret_history.append(LSTM_regret)
        
        if ep % 500 == 0:
            print(f'ep {ep:6d}')
            print(f'LSTM total reward: {LSTM_total_reward:5.2f} | DisRNN total reward: {DisRNN_total_reward:5.2f}')
            print_bottleneck_parameters(DisRNN)

        if ep > 0 and ep % 10_000 == 0:
            plot_regret_history(
                smooth(np.array(DisRNN_regret_history)), 
                smooth(np.array(LSTM_regret_history)),
                plot_name= 'training_regret_phase1'
            )
            torch.save({
                'ep': ep,
                'DisRNN_state_dict': DisRNN.state_dict(),
                'DisRNN_critic_state_dict': DisRNN_critic.state_dict(),
                'DisRNN_optimizer_state_dict': DisRNN_optimizer.state_dict(),
                'DisRNN_regret_history': DisRNN_regret_history,
                'LSTM_state_dict': LSTM.state_dict(),
                'LSTM_readout_state_dict': LSTM_readout.state_dict(),
                'LSTM_critic_state_dict': LSTM_critic.state_dict(),
                'LSTM_optimizer_state_dict': LSTM_optimizer.state_dict(),
                'LSTM_regret_history': LSTM_regret_history
            }, checkpoints_path + f'checkpoint_ep{ep}.pt')

        ep += 1


    disentanglement_ep = ep

    # display bottleneck parameters and plot regret histories at DisRNN disentanglement
    print_bottleneck_parameters(DisRNN)
    plot_regret_history(
        smooth(np.array(DisRNN_regret_history)), 
        smooth(np.array(LSTM_regret_history)),
        plot_name= 'training_regret_phase1'
    )
    torch.save({
        'ep': ep,
        'DisRNN_state_dict': DisRNN.state_dict(),
        'DisRNN_critic_state_dict': DisRNN_critic.state_dict(),
        'DisRNN_optimizer_state_dict': DisRNN_optimizer.state_dict(),
        'DisRNN_regret_history': DisRNN_regret_history,
        'LSTM_state_dict': LSTM.state_dict(),
        'LSTM_readout_state_dict': LSTM_readout.state_dict(),
        'LSTM_critic_state_dict': LSTM_critic.state_dict(),
        'LSTM_optimizer_state_dict': LSTM_optimizer.state_dict(),
        'LSTM_regret_history': LSTM_regret_history
    }, checkpoints_path + f'disentanglement_at_ep{ep}.pt')




    # --- Phase 2: search for best post-disentanglement model ------
    DisRNN_best_regret = np.inf
    LSTM_best_regret = np.inf

    eval_interval = config['eval_interval']
    eval_episodes = config['eval_episodes']
    search_episodes = config['search_episodes']
    for search_ep in range(search_episodes):
        DisRNN_regret, DisRNN_total_reward, LSTM_regret, LSTM_total_reward = run_training_episode(phase= 2, train_LSTM= True)
        DisRNN_regret_history.append(DisRNN_regret)
        LSTM_regret_history.append(LSTM_regret)

        if search_ep % eval_interval == 0:
            DisRNN_eval_regrets = []
            LSTM_eval_regrets = []
            for _ in range(eval_episodes):
                DisRNN_eval_regret, LSTM_eval_regret = run_eval_episode()
                DisRNN_eval_regrets.append(DisRNN_eval_regret)
                LSTM_eval_regrets.append(LSTM_eval_regret)

            DisRNN_cur_regret = np.mean(DisRNN_eval_regrets)
            if DisRNN_cur_regret < DisRNN_best_regret:
                DisRNN_best_regret = DisRNN_cur_regret
                torch.save({
                    'DisRNN_state_dict': DisRNN.state_dict()
                }, checkpoints_path + 'best_DisRNN.pt')

                print(f'ep {ep:6d}')
                print(f'LSTM total reward: {LSTM_total_reward:5.2f} | DisRNN total reward: {DisRNN_total_reward:5.2f}')
                print_bottleneck_parameters(DisRNN)
                
            LSTM_cur_regret = np.mean(LSTM_eval_regrets)
            if LSTM_cur_regret < LSTM_best_regret:
                LSTM_best_regret = LSTM_cur_regret
                torch.save({
                    'LSTM_state_dict': LSTM.state_dict(),
                    'LSTM_readout_state_dict': LSTM_readout.state_dict()
                }, checkpoints_path + 'best_LSTM.pt')

            plot_regret_history(
                smooth(np.array(DisRNN_regret_history[disentanglement_ep:])), 
                smooth(np.array(LSTM_regret_history[train_LSTM_until_ep:])),
                plot_name= 'training_regret_phase2'
            )

        ep += 1




for exp, config in exps.items():
    checkpoints_path = f'checkpoints/{exp}/seed{seed}/'
    figs_path = f'figs/{exp}/seed{seed}/'

    os.makedirs(checkpoints_path, exist_ok= True)
    os.makedirs(figs_path, exist_ok= True)

    train(config, checkpoints_path, figs_path)
