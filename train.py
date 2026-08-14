from config import *


def train(config, checkpoint_path, checkpoints_dir, figs_dir):
    # build env
    env = config['env']
    env.set_batch_size(config['batch_size'])


    # training hyperparameters
    lrs = config['lr']
    gammas = config['gamma']
    steps_unrolled = config['steps_unrolled']
    beta_e_annealed = config['beta_e_annealed']
    if beta_e_annealed:
        beta_e_floor = config['beta_e']
        anneal_end = 5000
    else:
        beta_e = config['beta_e']
    beta_v = config['beta_v']
    betas = config['beta']

    m_min = torch.logit(torch.tensor(0.01)).item()
    sigma_min = torch.log(torch.tensor(0.01)).item()

    eval_interval = config['eval_interval']
    eval_episodes = config['eval_episodes']
    search_episodes = config['search_episodes']


    # initialize training models and optimizers
    colors = config['color']

    def DisRNN_constructor():
        input_size = config['input_size']['DisRNN']
        hidden_size = config['hidden_size']['DisRNN']
        output_size = config['output_size']['DisRNN']
        model = MyDisRNN(hidden_size, input_size, output_size).to(device)
        critic = torch.nn.Linear(hidden_size, 1).to(device)
        parameters = list(model.parameters()) + list(critic.parameters())
        optimizer = torch.optim.Adam(parameters, lr= lrs['DisRNN'])
        return {
            'name': 'DisRNN',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'model': model,
            'critic': critic,
            'parameters': parameters,
            'optimizer': optimizer,
            'bottlenecks': ['h', 'x', 'z'],
            'beta': betas['DisRNN'],
            'converged': False
        }

    def DisLRU_constructor():
        input_size = config['input_size']['DisLRU']
        hidden_size = config['hidden_size']['DisLRU']
        output_size = config['output_size']['DisLRU']
        model = MyDisLRU(hidden_size, input_size, output_size).to(device)
        critic = torch.nn.Linear(hidden_size, 1).to(device)
        parameters = list(model.parameters()) + list(critic.parameters())
        optimizer = torch.optim.Adam(parameters, lr= lrs['DisLRU'])
        return {
            'name': 'DisLRU',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'model': model,
            'critic': critic,
            'parameters': parameters,
            'optimizer': optimizer,
            'bottlenecks': ['x', 'z'],
            'beta': betas['DisLRU'],
            'converged': False
        }

    def LSTM_constructor():
        input_size = config['input_size']['LSTM']
        hidden_size = config['hidden_size']['LSTM']
        output_size = config['output_size']['LSTM']
        model = torch.nn.LSTM(input_size, hidden_size).to(device)
        readout = torch.nn.Linear(hidden_size, output_size).to(device)
        critic = torch.nn.Linear(hidden_size, 1).to(device)
        parameters = list(model.parameters()) + list(readout.parameters()) + list(critic.parameters())
        optimizer = torch.optim.Adam(parameters, lr= lrs['LSTM'])
        return {
            'name': 'LSTM',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'model': model,
            'readout': readout,
            'critic': critic,
            'parameters': parameters,
            'optimizer': optimizer,
            'train_until_ep': config['train_until_ep']['LSTM']
        }

    model_constructors = {
        'DisRNN': DisRNN_constructor,
        'DisLRU': DisLRU_constructor,
        'LSTM': LSTM_constructor
    }
    models = {
        model: model_constructors[model]() 
        for model in config['models']
    }

    env.build_models(models)


    # --- training ---

    prev_state_dicts = {model: copy.deepcopy(models[model]['model'].state_dict()) for model in models}
    training_phase = {model: 1 for model in models}
    regret_histories = {'phase1': {model: [] for model in models}, 'phase2': {}}
    best_regret = {model: np.inf for model in models}
    cur_regret = {model: None for model in models}
    ep = 0

    # load checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location= device)
        ep, prev_state_dicts, training_phase, regret_histories, models = load_checkpoint(checkpoint, models)
        ep += 1
 
    active_models = {model for model in models if training_phase[model] != 0}
    while active_models:
        # reset active models
        for model in active_models:
            models[model]['model'].train()
            models[model]['optimizer'].zero_grad()
        env.reset(active_models)

        # run training episode
        rewards, expected_returns, log_probs, entropies, regrets, bottleneck_losses = env.training_episode(active_models, steps_unrolled)
        total_rewards = {model: rewards[model].sum(dim= 0).mean().item() for model in active_models}
        avg_regrets = {model: regrets[model].mean().item() for model in active_models}
        for model in active_models:
            regret_histories[f'phase{training_phase[model]}'][model].append(avg_regrets[model])

        # evaluate models in phase 2
        phase2_models = {model for model in models if training_phase[model] == 2}
        for model in phase2_models:
            search_ep = len(regret_histories['phase2'][model])
            if search_ep >= search_episodes:
                training_phase[model] = 0  # end training
            else:
                if search_ep % eval_interval == 0:
                    with torch.no_grad():
                        models[model]['model'].eval()
                        
                        eval_regrets = []
                        for _ in range(eval_episodes):
                            env.reset({model})

                            eval_regret = env.eval_episode({model})[model].mean().item()
                            eval_regrets.append(eval_regret)
            
                        # save best model if current average regret is better than previous best regret
                        cur_regret[model] = np.mean(eval_regrets)
                        if cur_regret[model] < best_regret[model]:
                            best_regret[model] = cur_regret[model]
                            save_dict = {f'{model}_state_dict': models[model]['model'].state_dict()}
                            if 'readout' in models[model]:
                                save_dict[f'{model}_readout_state_dict'] = models[model]['readout'].state_dict()
                            torch.save(save_dict, checkpoints_dir + f'best_{model}.pt')
        
        
        # --- advantage actor-critic ------

        # update exploration parameter
        if beta_e_annealed:
            beta_e = beta_e_floor + (1.0 - beta_e_floor) * max(0.0, 1.0 - ep / anneal_end)

        # gradient descent
        for model in active_models:
            returns = rewards[model].clone()
            for t in reversed(range(len(returns) - 1)):
                returns[t] = rewards[model][t] + gammas[model] * returns[t+1]
            returns = (returns - returns.mean(dim= 1, keepdim= True)) / (returns.std(dim= 1, keepdim= True) + 1e-8)
            advantage = returns - expected_returns[model]

            # update bottleneck pressure
            if 'beta' in models[model]:
                floor = models[model]['beta']['floor']
                ceil = models[model]['beta']['ceil']
                warmup_start = models[model]['beta']['warmup']['start']
                warmup_end = models[model]['beta']['warmup']['end']
                if ep < warmup_start or training_phase[model] == 2:
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


        # print episode rewards and bottleneck parameters, convergence check
        if ep % 2500 == 0:
            print(f'ep {ep}')
            print(" | ".join(f"{model} total reward: {total_rewards[model]:5.2f}" for model in active_models))
            for model in active_models:
                if 'bottlenecks' in models[model]: 
                    print_bottleneck_parameters(models[model])
                    models[model]['converged'] = bottlenecks_converged(models[model], prev_state_dicts[model])
                    prev_state_dicts[model] = copy.deepcopy(models[model]['model'].state_dict())

        # update training status
        for model in active_models:
            if training_phase[model] == 1:
                if 'bottlenecks' in models[model]:
                    if ep >= models[model]['beta']['warmup']['end'] and (disentangled(models[model]) or models[model]['converged']):
                        training_phase[model] = 2
                        regret_histories['phase2'][model] = []
                        torch.save(build_checkpoint(
                            ep, 
                            prev_state_dicts, 
                            training_phase, 
                            regret_histories, 
                            models
                        ), checkpoints_dir + f'{model}_disentanglement_at_ep{ep}.pt')  # save disentanglement checkpoint
                else:
                    if ep >= models[model]['train_until_ep']:
                        training_phase[model] = 2
                        regret_histories['phase2'][model] = []

        # plot regret histories, save checkpoint
        if ep > 0 and ep % 10_000 == 0:
            plot_regret_histories(regret_histories, figs_dir, colors)
            torch.save(build_checkpoint(
                ep, 
                prev_state_dicts, 
                training_phase, 
                regret_histories, 
                models
            ), checkpoints_dir + f'checkpoint_ep{ep}.pt')

        ep += 1
        active_models = {model for model in models if training_phase[model] != 0}
        

    # print final bottleneck parameters
    for model in models:
        if 'bottlenecks' in models[model]: 
            print_bottleneck_parameters(models[model])
