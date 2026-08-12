from config import *
from helpers import format_matrix, smooth


def train(config, checkpoint_path, checkpoints_dir, figs_dir):
    # set task
    D = config['D']
    num_trials = config['num_trials']
    restless = config['restless']
    drift = config['drift']
    dependent_arms = config['dependent_arms']


    # initialize training models and optimizers
    def DisRNN_constructor():
        model = MyDisRNN(config['hidden_size']['DisRNN'], config['input_size']['DisRNN'], num_arms).to(device)
        critic = torch.nn.Linear(config['hidden_size']['DisRNN'], 1).to(device)
        parameters = list(model.parameters()) + list(critic.parameters())
        optimizer = torch.optim.Adam(parameters, lr= config['lr']['DisRNN'])
        return {
            'model': model,
            'critic': critic,
            'parameters': parameters,
            'optimizer': optimizer,
            'bottlenecks': ['h', 'x', 'z'],
            'beta': config['beta']['DisRNN'],
            'converged': False,
            'disentangled': False,
            'disentanglement_ep': 0,
            'color': config['colors']['DisRNN'],
            'linestyle': config['linestyles']['DisRNN']
        }

    def DisLRU_constructor():
        model = MyDisLRU(config['hidden_size']['DisLRU'], config['input_size']['DisLRU'], num_arms).to(device)
        critic = torch.nn.Linear(config['hidden_size']['DisLRU'], 1).to(device)
        parameters = list(model.parameters()) + list(critic.parameters())
        optimizer = torch.optim.Adam(parameters, lr= config['lr']['DisLRU'])
        return {
            'model': model,
            'critic': critic,
            'parameters': parameters,
            'optimizer': optimizer,
            'bottlenecks': ['x', 'z'],
            'beta': config['beta']['DisLRU'],
            'converged': False,
            'disentangled': False,
            'disentanglement_ep': 0,
            'color': config['colors']['DisLRU'],
            'linestyle': config['linestyles']['DisLRU']
        }

    def LSTM_constructor():
        model = torch.nn.LSTM(config['input_size']['LSTM'], config['hidden_size']['LSTM']).to(device)
        readout = torch.nn.Linear(config['hidden_size']['LSTM'], num_arms).to(device)
        critic = torch.nn.Linear(config['hidden_size']['LSTM'], 1).to(device)
        parameters = list(model.parameters()) + list(readout.parameters()) + list(critic.parameters())
        optimizer = torch.optim.Adam(parameters, lr= config['lr']['LSTM'])
        return {
            'model': model,
            'readout': readout,
            'critic': critic,
            'parameters': parameters,
            'optimizer': optimizer,
            'train_until_ep': config['train_until_ep']['LSTM'],
            'color': config['colors']['LSTM'],
            'linestyle': config['linestyles']['LSTM']
        }

    model_constructors = {
        'DisRNN': DisRNN_constructor,
        'DisLRU': DisLRU_constructor,
        'LSTM': LSTM_constructor
    }
    models = {
        model: model_constructors[model]() 
        for model in ['DisLRU', 'LSTM']
    }

    m_min = torch.logit(torch.tensor(0.01)).item()
    sigma_min = torch.log(torch.tensor(0.01)).item()


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


    # training helpers
    def plot_regret_history(regret_histories, plot_name):
        plt.figure(figsize= (8,5))
        for model, history in regret_histories.items():
            plt.plot(history, label= model, color= models[model]['color'])
        plt.xlabel('Episode')
        plt.ylabel('Regret')
        plt.title('Model Regret Over Time')
        plt.legend()
        plt.grid()
        plt.savefig(figs_dir + f'{plot_name}.png')
        plt.close()


    def print_bottleneck_parameters(model):
        with torch.no_grad():
            print()
            print(model)
            print()
            for bottleneck in models[model]['bottlenecks']:
                m = torch.sigmoid(getattr(models[model]['model'], f'logit_M_{bottleneck}'))
                print(format_matrix(m, f'M_{bottleneck}', row_prefix= 'lat', col_prefix= 'lat'))
                print()
            print()


    def build_checkpoint(ep):
        checkpoint = {'ep': ep, 'prev_state_dicts': prev_state_dicts}
        for model in models:
            checkpoint[f'{model}_state_dict'] = models[model]['model'].state_dict()
            checkpoint[f'{model}_critic_state_dict'] = models[model]['critic'].state_dict()
            checkpoint[f'{model}_optimizer_state_dict'] = models[model]['optimizer'].state_dict()
            checkpoint[f'{model}_regret_history'] = regret_histories[model]
            if 'readout' in models[model]:
                checkpoint[f'{model}_readout_state_dict'] = models[model]['readout'].state_dict()
        
        return checkpoint


    def disentangled(model, low= 0.1, high= 0.9):
        with torch.no_grad():
            checks = []
            for bottleneck in models[model]['bottlenecks']:
                m = torch.sigmoid(getattr(models[model]['model'], f'logit_M_{bottleneck}'))
                sigma = torch.exp(getattr(models[model]['model'], f'log_sigma_{bottleneck}'))
                checks.append(((m <= low) | (m >= high)).all() and ((sigma <= low) | (sigma >= high)).all())

            return bool(all(checks))


    def bottlenecks_converged(model, prev_state_dicts, tol= 0.02):
        with torch.no_grad():
            checks = []
            for bottleneck in models[model]['bottlenecks']:
                prev_m = torch.sigmoid(prev_state_dicts[model][f'logit_M_{bottleneck}'])
                prev_sigma = torch.exp(prev_state_dicts[model][f'log_sigma_{bottleneck}'])
                m = torch.sigmoid(getattr(models[model]['model'], f'logit_M_{bottleneck}'))
                sigma = torch.exp(getattr(models[model]['model'], f'log_sigma_{bottleneck}'))
                checks.append((torch.abs(m - prev_m) < tol).all() and (torch.abs(sigma - prev_sigma) < tol).all())

            return bool(all(checks))


    def run_training_episode(active_models, phase= 1):
        # sample task
        probs = D(batch_size, num_arms, device)

        # activate training mode and reset gradients
        for model in active_models:
            models[model]['model'].train()
            models[model]['optimizer'].zero_grad()

        # reset model states
        h, c, x = {}, {}, {}
        if 'DisRNN' in active_models:
            h['DisRNN'] = torch.zeros(batch_size, config['hidden_size']['DisRNN'], device= device)
            x['DisRNN'] = torch.zeros(batch_size, config['input_size']['DisRNN'], device= device)
        if 'DisLRU' in active_models:
            h['DisLRU'] = torch.zeros(batch_size, config['hidden_size']['DisLRU'], device= device)
            x['DisLRU'] = torch.zeros(batch_size, config['input_size']['DisLRU'], device= device)
        if 'LSTM' in active_models:
            h['LSTM'] = torch.zeros(1, batch_size, config['hidden_size']['LSTM'], device= device)
            c['LSTM'] = torch.zeros(1, batch_size, config['hidden_size']['LSTM'], device= device)
            x['LSTM'] = torch.zeros(batch_size, config['input_size']['LSTM'], device= device)

        log_probs = {model: [] for model in active_models}
        rewards = {model: [] for model in active_models}
        expected_returns = {model: [] for model in active_models}
        entropies = {model: [] for model in active_models}
        regrets = {model: [] for model in active_models}
        bottleneck_losses = {
            model: {bottleneck: [] for bottleneck in models[model]['bottlenecks']} 
            for model in active_models if 'bottlenecks' in models[model]
        }

        for t in range(num_trials):
            optimal = probs.max(dim= -1).values

            # detach gradients
            if t % steps_unrolled == 0:
                for model in h: 
                    h[model] = h[model].detach() 
                for model in c: 
                    c[model] = c[model].detach() 

            # step
            logits = {model: None for model in active_models}
            kls = {
                model: {bottleneck: None for bottleneck in models[model]['bottlenecks']} 
                for model in active_models if 'bottlenecks' in models[model]
            }
            critic_inputs = {model: None for model in active_models}

            if 'DisRNN' in active_models:
                h['DisRNN'], kls['DisRNN'] = models['DisRNN']['model'].step(h['DisRNN'], x['DisRNN'])
                logits['DisRNN'] = models['DisRNN']['model'].out(h['DisRNN'])
                critic_inputs['DisRNN'] = h['DisRNN'].detach()
            if 'DisLRU' in active_models:
                h['DisLRU'], kls['DisLRU'] = models['DisLRU']['model'].step(h['DisLRU'], x['DisLRU'])
                logits['DisLRU'] = models['DisLRU']['model'].out(h['DisLRU'])
                critic_inputs['DisLRU'] = h['DisLRU'].detach()
            if 'LSTM' in active_models:
                out, (h['LSTM'], c['LSTM']) = models['LSTM']['model'](x['LSTM'].unsqueeze(0), (h['LSTM'], c['LSTM']))
                logits['LSTM'] = models['LSTM']['readout'](out.squeeze(0))
                critic_inputs['LSTM'] = out.squeeze(0)

            # sample
            a = {model: None for model in active_models}
            r = {model: None for model in active_models}
            for model in active_models:
                pi = torch.distributions.Categorical(logits= logits[model])
                a[model] = pi.sample()
                r[model] = (torch.rand(batch_size, device= device) < probs[batch_idx, a[model]]).float()

                log_probs[model].append(pi.log_prob(a[model]))
                rewards[model].append(r[model])
                entropies[model].append(pi.entropy())
                expected_returns[model].append(models[model]['critic'](critic_inputs[model]).squeeze(-1))
                regrets[model].append(optimal - probs[batch_idx, a[model]])
                if model in bottleneck_losses:
                    for key, val in kls[model].items():
                        bottleneck_losses[model][key].append(val)

            # update obs
            if 'DisRNN' in active_models:
                x['DisRNN'] = torch.stack([2*a['DisRNN'].float() - 1, 2*r['DisRNN'] - 1], dim= -1)
            if 'DisLRU' in active_models:
                x['DisLRU'] = torch.zeros(batch_size, config['input_size']['DisLRU'], device= device)
                x['DisLRU'][torch.arange(batch_size, device= device), a['DisLRU']] = 2*r['DisLRU'] - 1
            if 'LSTM' in active_models:
                x['LSTM'] = torch.stack([2*a['LSTM'].float() - 1, 2*r['LSTM'] - 1], dim= -1)

            # restless bandits
            if restless:
                probs += drift * torch.randn(batch_size, num_arms, device= device)
                probs = torch.clamp(probs, 0, 1)
                if dependent_arms:
                    probs[:, 1] = 1 - probs[:, 0]

        log_probs = {model: torch.stack(vals) for model, vals in log_probs.items()}
        rewards = {model: torch.stack(vals) for model, vals in rewards.items()}
        expected_returns = {model: torch.stack(vals) for model, vals in expected_returns.items()}
        entropies = {model: torch.stack(vals) for model, vals in entropies.items()}
        regrets = {model: torch.stack(vals) for model, vals in regrets.items()}
        bottleneck_losses = {
            model: {key: torch.stack(vals) for key, vals in bottleneck_losses[model].items()}
            for model in bottleneck_losses
        }

        
        # --- advantage actor-critic ------

        # update exploration parameter
        if beta_e_annealed:
            beta_e = beta_e_floor + (1.0 - beta_e_floor) * max(0.0, 1.0 - ep / anneal_end)

        # gradient descent
        for model in active_models:
            returns = rewards[model].clone()
            for t in reversed(range(num_trials - 1)):
                returns[t] = rewards[model][t] + config['gamma'][model] * returns[t+1]
            returns = (returns - returns.mean(dim= 1, keepdim= True)) / (returns.std(dim= 1, keepdim= True) + 1e-8)
            advantage = returns - expected_returns[model]

            # update bottleneck pressure
            if model in config['beta']:
                floor = models[model]['beta']['floor']
                ceil = models[model]['beta']['ceil']
                warmup_start = models[model]['beta']['warmup']['start']
                warmup_end = models[model]['beta']['warmup']['end']
                if ep < warmup_start or phase == 2:
                    beta = floor
                else:
                    beta = floor + (ceil - floor) * min((ep - warmup_start) / (warmup_end - warmup_start), 1.0)
            else:
                beta = 0.0

            # calculate loss
            loss_actor = -(log_probs[model] * advantage.detach()).mean()
            loss_critic = torch.nn.functional.mse_loss(expected_returns[model], returns)
            loss_entropy = entropies[model].mean()
            loss_bottlenecks = sum(loss.mean() for loss in bottleneck_losses[model].values()) if model in bottleneck_losses else 0.0
            loss = loss_actor + beta_v * loss_critic - beta_e * loss_entropy + beta * loss_bottlenecks

            # take gradient step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[model]['parameters'], max_norm= 1.0)
            models[model]['optimizer'].step()

            # maintain bottleneck parameters
            if model in bottleneck_losses:
                with torch.no_grad():
                    for bottleneck in models[model]['bottlenecks']:
                        getattr(models[model]['model'], f'logit_M_{bottleneck}').clamp_(min= m_min)
                        getattr(models[model]['model'], f'log_sigma_{bottleneck}').clamp_(min= sigma_min, max= 0.0)

        return {
            'regret': {model: regrets[model].mean().item() for model in active_models},
            'reward': {model: rewards[model].sum(dim= 0).mean().item() for model in active_models}
        }


    def run_eval_episode():
        with torch.no_grad():
            # sample task
            probs = D(batch_size, num_arms, device)

            # activate eval mode
            for model in models:
                models[model]['model'].eval()

            # reset model states
            h, c, x = {}, {}, {}
            if 'DisRNN' in models:
                h['DisRNN'] = torch.zeros(batch_size, config['hidden_size']['DisRNN'], device= device)
                x['DisRNN'] = torch.zeros(batch_size, config['input_size']['DisRNN'], device= device)
            if 'DisLRU' in models:
                h['DisLRU'] = torch.zeros(batch_size, config['hidden_size']['DisLRU'], device= device)
                x['DisLRU'] = torch.zeros(batch_size, config['input_size']['DisLRU'], device= device)
            if 'LSTM' in models:
                h['LSTM'] = torch.zeros(1, batch_size, config['hidden_size']['LSTM'], device= device)
                c['LSTM'] = torch.zeros(1, batch_size, config['hidden_size']['LSTM'], device= device)
                x['LSTM'] = torch.zeros(batch_size, config['input_size']['LSTM'], device= device)

            regrets = {model: [] for model in models}

            for t in range(num_trials):
                optimal = probs.max(dim= -1).values

                # step
                logits = {model: None for model in models}
                if 'DisRNN' in models:
                    h['DisRNN'], _ = models['DisRNN']['model'].step(h['DisRNN'], x['DisRNN'])
                    logits['DisRNN'] = models['DisRNN']['model'].out(h['DisRNN'])
                if 'DisLRU' in models:
                    h['DisLRU'], _ = models['DisLRU']['model'].step(h['DisLRU'], x['DisLRU'])
                    logits['DisLRU'] = models['DisLRU']['model'].out(h['DisLRU'])
                if 'LSTM' in models:
                    out, (h['LSTM'], c['LSTM']) = models['LSTM']['model'](x['LSTM'].unsqueeze(0), (h['LSTM'], c['LSTM']))
                    logits['LSTM'] = models['LSTM']['readout'](out.squeeze(0))

                # sample
                a = {model: None for model in models}
                r = {model: None for model in models}
                for model in models:
                    pi = torch.distributions.Categorical(logits= logits[model])
                    a[model] = pi.sample()
                    r[model] = (torch.rand(batch_size, device= device) < probs[batch_idx, a[model]]).float()
                    regrets[model].append(optimal - probs[batch_idx, a[model]])

                # update obs
                if 'DisRNN' in models:
                    x['DisRNN'] = torch.stack([2*a['DisRNN'].float() - 1, 2*r['DisRNN'] - 1], dim= -1)
                if 'DisLRU' in models:
                    x['DisLRU'] = torch.zeros(batch_size, config['input_size']['DisLRU'], device= device)
                    x['DisLRU'][torch.arange(batch_size, device= device), a['DisLRU']] = 2*r['DisLRU'] - 1
                if 'LSTM' in models:
                    x['LSTM'] = torch.stack([2*a['LSTM'].float() - 1, 2*r['LSTM'] - 1], dim= -1)

                # restless bandits
                if restless:
                    probs += drift * torch.randn(batch_size, num_arms, device= device)
                    probs = torch.clamp(probs, 0, 1)
                    if dependent_arms:
                        probs[:, 1] = 1 - probs[:, 0]

            regrets = {model: torch.stack(vals) for model, vals in regrets.items()}

            return {'regret': {model: regrets[model].mean().item() for model in models}}




    # --- Phase 1: train until disentanglement ------
    regret_histories = {model: [] for model in models}
    prev_state_dicts = {model: copy.deepcopy(models[model]['model'].state_dict()) for model in models}
    training = {model: True for model in models}
    was_training = {model: True for model in models}

    ep = 0

    # load checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location= device)
        ep = checkpoint['ep']
        prev_state_dicts = checkpoint['prev_state_dicts']
        for model in models:
            models[model]['model'].load_state_dict(checkpoint[f'{model}_state_dict'])
            models[model]['critic'].load_state_dict(checkpoint[f'{model}_critic_state_dict'])
            models[model]['optimizer'].load_state_dict(checkpoint[f'{model}_optimizer_state_dict'])
            regret_histories[model] = checkpoint[f'{model}_regret_history']
            if 'readout' in models[model]:
                models[model]['readout'].load_state_dict(checkpoint[f'{model}_readout_state_dict'])
            if 'bottlenecks' in models[model]:
                models[model]['converged'] = bottlenecks_converged(model, prev_state_dicts)
                models[model]['disentanglement_ep'] = len(regret_histories[model])

    # get training status
    for model in models: 
        if 'bottlenecks' in models[model]: 
            training[model] = ep < models[model]['beta']['warmup']['end'] or (not disentangled(model) and not models[model]['converged'])
        else:
            training[model] = ep < models[model]['train_until_ep']

    # phase 1 training
    while any(training[model] for model in training):
        # run training episode for models still training
        active_models = {model for model in training if training[model]}
        training_ep_res = run_training_episode(active_models, phase= 1)
        for model in active_models:
            regret_histories[model].append(training_ep_res['regret'][model])

        # print episode rewards and model bottleneck parameters
        if ep % 500 == 0:
            print(f'ep {ep:6d}')
            print(" | ".join(f"{model} total reward: {training_ep_res['reward'][model]:5.2f}" for model in active_models))
            for model in active_models:
                if 'bottlenecks' in models[model]: 
                    print_bottleneck_parameters(model)

        # update phase 1 regret history plot and save checkpoint
        if ep > 0 and ep % 10_000 == 0:
            plot_regret_history(
                {model: smooth(np.array(regret_histories[model])) for model in models}, 
                plot_name= 'training_regret_phase1'
            )
            torch.save(build_checkpoint(ep), checkpoints_dir + f'checkpoint_ep{ep}.pt')

            # check for bottleneck parameter convergence
            for model in models: 
                if 'bottlenecks' in models[model]: 
                    models[model]['converged'] = bottlenecks_converged(model, prev_state_dicts)
                    prev_state_dicts[model] = copy.deepcopy(models[model]['model'].state_dict())

        # update training status
        ep += 1
        for model in models: 
            if 'bottlenecks' in models[model]:
                was_training[model] = training[model]
                training[model] = ep < models[model]['beta']['warmup']['end'] or (not disentangled(model) and not models[model]['converged'])
                # save disentanglement checkpoint if model disentangled or bottleneck parameters converged (after beta warmup)
                if was_training[model] and not training[model]:
                    models[model]['disentanglement_ep'] = ep
                    torch.save(build_checkpoint(ep), checkpoints_dir + f'{model}_disentanglement_at_ep{ep}.pt')
            else:
                training[model] = ep < models[model]['train_until_ep']

    # print final bottleneck parameters and finalize phase 1 regret history plot
    phase_2_start_eps = {model: None for model in models}
    for model in models:
        if 'bottlenecks' in models[model]: 
            print_bottleneck_parameters(model)
            phase_2_start_eps[model] = models[model]['disentanglement_ep']
        else:
            phase_2_start_eps[model] = models[model]['train_until_ep']
    plot_regret_history(
        {model: smooth(np.array(regret_histories[model])) for model in models}, 
        plot_name= 'training_regret_phase1'
    )


    # --- Phase 2: search for best post-disentanglement model ------
    best_regrets = {model: np.inf for model in models}
    cur_regrets = {model: None for model in models}

    eval_interval = config['eval_interval']
    eval_episodes = config['eval_episodes']
    search_episodes = config['search_episodes']
    for search_ep in range(search_episodes):
        # run training episode for all models (min bottleneck pressure)
        training_ep_res = run_training_episode(models, phase= 2)
        for model in models:
            regret_histories[model].append(training_ep_res['regret'][model])

        # evaluate models at interval
        if search_ep % eval_interval == 0:
            # get regret data for model evaluation
            eval_regrets = {model: [] for model in models}
            for _ in range(eval_episodes):
                eval_ep_res = run_eval_episode()
                for model in models:
                    eval_regrets[model].append(eval_ep_res['regret'][model])

            # save best model if current average regret is better than previous best regret
            for model in models:
                cur_regrets[model] = np.mean(eval_regrets[model])
                if cur_regrets[model] < best_regrets[model]:
                    best_regrets[model] = cur_regrets[model]

                    save_dict = {f'{model}_state_dict': models[model]['model'].state_dict()}
                    if 'readout' in models[model]:
                        save_dict[f'{model}_readout_state_dict'] = models[model]['readout'].state_dict()
                    torch.save(save_dict, checkpoints_dir + f'best_{model}.pt')

            # update phase 2 regret history plot
            plot_regret_history(
                {model: smooth(np.array(regret_histories[model][phase_2_start_eps[model]:])) for model in models}, 
                plot_name= 'training_regret_phase2'
            )

        ep += 1




def main():
    for exp, config in exps.items():
        checkpoints_dir = f'checkpoints/{exp}/seed{seed}/'
        figs_dir = f'figs/{exp}/seed{seed}/'

        train_res = input(f"Begin training for experiment: {exp}, seed {seed}? (y/n): ")
        if train_res.lower() == 'n':
            continue

        resume_checkpoint = False
        checkpoint_path = ''

        if os.path.exists(checkpoints_dir) or os.path.exists(figs_dir):
            overwrite_res = input(f"There is history for this experiment. Continue? (y/n): ")
            if overwrite_res.lower() == 'n':
                continue

            resume_checkpoint_res = input(f"Do you want to resume training from a checkpoint? (y/n): ")
            resume_checkpoint = resume_checkpoint_res.lower() == 'y'

            if resume_checkpoint:
                while True:
                    checkpoint_path = input("Resume checkpoint path: ")
                    if os.path.exists(checkpoint_path):
                        break

                    cont = input(f"Cannot find checkpoint: {checkpoint_path}. Enter another checkpoint? (y/n): ")
                    if cont.lower() != 'y':
                        resume_checkpoint = False
                        checkpoint_path = ''
                        break
                                
        os.makedirs(checkpoints_dir, exist_ok= True)
        os.makedirs(figs_dir, exist_ok= True)

        if resume_checkpoint:
            print(f"Resuming training for experiment {exp} bandits, seed {seed} from checkpoint {checkpoint_path}.\n")
        else:
            print(f"Beginning training for experiment {exp} bandits, seed {seed}.\n")
            
        train(config, checkpoint_path, checkpoints_dir, figs_dir)


if __name__ == "__main__":
    main()
