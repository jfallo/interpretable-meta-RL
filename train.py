import torch
import numpy as np

from helpers import model_constructor, load_checkpoint, build_checkpoint, print_bottleneck_parameters, plot_regret_histories


def train(config, figs_dir, checkpoints_dir, checkpoint_path= None):
    # build env
    env = config['env']['env'](config= config['env']['config'], 
                               device= config['env']['device'])
    env.set_batch_size(config['batch_size'])
    device = config['env']['device']

    # training hyperparameters
    lrs = config['lr']
    gammas = config['gamma']
    steps_unrolled = config['steps_unrolled']
    beta_v = config['beta_v']
    beta_e_ceil = config['beta_e']['ceil']
    beta_e_floor = config['beta_e']['floor']
    anneal_start = config['beta_e']['anneal']['start']
    anneal_end = config['beta_e']['anneal']['end']
    tau = config['beta_e']['tau']

    m_min = torch.logit(torch.tensor(0.01)).item()
    sigma_min = torch.log(torch.tensor(0.01)).item()

    # initialize training models and optimizers
    models = {model: model_constructor(model, config, device, training= True, lr= lrs[model]) 
              for model in config['models']}
    env.build_models(models)


    # --- training --- #

    regret_histories = {model: [] for model in models}
    ep = 0

    # load checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location= device)
        ep, regret_histories, models = load_checkpoint(checkpoint, models)
        ep += 1
 
    active_models = sorted(model for model in models if ep <= models[model]['training_eps'])
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
            regret_histories[model].append(avg_regrets[model])
                
        
        # --- advantage actor-critic ----

        # update exploration parameter
        if ep < anneal_start:
            beta_e = beta_e_ceil
        elif ep < anneal_end:
            beta_e = beta_e_floor + (beta_e_ceil - beta_e_floor) * np.exp(-(ep - anneal_start) / tau)
        else:
            beta_e = beta_e_floor

        # gradient descent
        for model in active_models:
            returns = rewards[model].clone()
            for t in reversed(range(len(returns) - 1)):
                returns[t] = rewards[model][t] + gammas[model] * returns[t+1]
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantage = returns - expected_returns[model]

            # update bottleneck pressure
            beta = 0.0
            if 'beta' in models[model]:
                beta_ceil = models[model]['beta']['ceil']
                warmup_start = models[model]['beta']['warmup']['start']
                warmup_end = models[model]['beta']['warmup']['end']
                if ep >= warmup_start:
                    beta = beta_ceil * min((ep - warmup_start) / (warmup_end - warmup_start), 1.0)

            # calculate loss
            loss_actor = -(log_probs[model] * advantage.detach()).mean()
            loss_critic = torch.nn.functional.mse_loss(expected_returns[model], returns)
            loss_entropy = entropies[model].mean()
            loss_bottlenecks = sum(loss.mean() for loss in bottleneck_losses[model].values()) if model in bottleneck_losses else 0.0
            loss = loss_actor + beta_v * loss_critic - beta_e * loss_entropy + beta * loss_bottlenecks

            # take gradient step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[model]['actor_parameters'], max_norm= 1.0)
            torch.nn.utils.clip_grad_norm_(models[model]['critic_parameters'], max_norm= 1.0)
            models[model]['optimizer'].step()

            # maintain bottleneck parameters
            if model in bottleneck_losses:
                with torch.no_grad():
                    for bottleneck in models[model]['bottlenecks']:
                        if hasattr(models[model]['model'], f'logit_M_{bottleneck}'):
                            getattr(models[model]['model'], f'logit_M_{bottleneck}').clamp_(min= m_min)
                        getattr(models[model]['model'], f'log_sigma_{bottleneck}').clamp_(min= sigma_min, max= 0.0)


        # print episode rewards and bottleneck parameters, plot regret histories
        if ep % 2500 == 0:
            print(f'ep {ep}')
            print(" | ".join(f"{model} total reward: {total_rewards[model]:5.2f}" for model in active_models))
            for model in active_models:
                if 'bottlenecks' in models[model]: 
                    print_bottleneck_parameters(models[model])
            plot_regret_histories(regret_histories, figs_dir, config['color'])

        # save checkpoint
        if ep > 0 and ep % 10_000 == 0:
            torch.save(build_checkpoint(ep, regret_histories, models), 
                       checkpoints_dir + f'checkpoint_ep{ep}.pt')

        # update training status
        active_models = sorted(model for model in models if ep < models[model]['training_eps'])
        ep += 1

    for model in models:
        if 'bottlenecks' in models[model]:
            print_bottleneck_parameters(models[model])
    plot_regret_histories(regret_histories, figs_dir, config['color'])
    