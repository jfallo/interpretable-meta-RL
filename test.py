from config import *

from agents.Thompson import Thompson
from agents.UCB import UCB
from agents.Gittins import compute_gittins_table, Gittins


def test(config, checkpoints_path, figs_path):
    # set task
    D = config['D']
    num_trials = config['num_trials']
    restless = config['restless']
    drift = config['drift']
    dependent_arms = config['dependent_arms']


    # build Gittins index table
    gittins_table = compute_gittins_table(max_total= num_trials+1, gamma= config['gamma']['Gittins'], N= 200, tol= 1e-4)
    
    # initialize models
    def DisRNN_constructor():
        model = MyDisRNN(config['hidden_size']['DisRNN'], config['input_size']['DisRNN'], num_arms).to(device)
        return {'model': model, 'color': config['colors']['DisRNN'], 'linestyle': config['linestyles']['DisRNN']}
    
    def DisLRU_constructor():
        model = MyDisLRU(config['hidden_size']['DisLRU'], config['input_size']['DisLRU'], num_arms).to(device)
        return {'model': model, 'color': config['colors']['DisLRU'], 'linestyle': config['linestyles']['DisLRU']}
    
    def LSTM_constructor():
        model = torch.nn.LSTM(config['input_size']['LSTM'], config['hidden_size']['LSTM']).to(device)
        readout = torch.nn.Linear(config['hidden_size']['LSTM'], num_arms).to(device)
        return {'model': model, 'readout': readout, 'color': config['colors']['LSTM'], 'linestyle': config['linestyles']['LSTM']}
    
    model_constructors = {
        'DisRNN': DisRNN_constructor,
        'DisLRU': DisLRU_constructor,
        'LSTM': LSTM_constructor
    }
    trained_models = {
        model: model_constructors[model]() 
        for model in ['DisRNN', 'DisLRU', 'LSTM']  # --- list of models to test ---
    }
    classical_models = {
        'Thompson': {'model': None, 'color': config['colors']['Thompson'], 'linestyle': config['linestyles']['Thompson']},
        'UCB': {'model': None, 'color': config['colors']['UCB'], 'linestyle': config['linestyles']['UCB']},
        'Gittins': {'model': None, 'color': config['colors']['Gittins'], 'linestyle': config['linestyles']['Gittins']}
    }
    models = {**trained_models, **classical_models}


    # testing helpers
    def plot_agent(data, color, linestyle, label, plot_std= False):
        mean = np.stack(data).mean(axis= 0)
        plt.plot(mean, color= color, linestyle= linestyle, label= label)
        if plot_std:
            std = np.stack(data).std(axis= 0, ddof= 1)
            plt.fill_between(range(num_trials), mean - std, mean + std, alpha= 0.1, color= color, linestyle= linestyle)


    def optimal_arm_rate(raw_regrets):
        regrets = np.stack(raw_regrets)
        return (regrets == 0).mean(axis= 0)


    def run_tests(num_tests):
        with torch.no_grad():
            raw_regrets = {model: [] for model in models}
            cumulative_regrets = {model: [] for model in models}
            for _ in range(num_tests):
                # sample task
                probs = D(1, num_arms, device)
                
                # activate eval mode
                for model in trained_models:
                    trained_models[model]['model'].eval()
                
                # reset model states
                h, c, x = {}, {}, {}
                if 'DisRNN' in models:
                    h['DisRNN'] = torch.zeros(1, config['hidden_size']['DisRNN'], device= device)
                    x['DisRNN'] = torch.zeros(1, config['input_size']['DisRNN'], device= device)
                if 'DisLRU' in models:
                    h['DisLRU'] = torch.zeros(1, config['hidden_size']['DisLRU'], device= device)
                    x['DisLRU'] = torch.zeros(1, config['input_size']['DisLRU'], device= device)
                if 'LSTM' in models:
                    h['LSTM'] = torch.zeros(1, 1, config['hidden_size']['LSTM'], device= device)
                    c['LSTM'] = torch.zeros(1, 1, config['hidden_size']['LSTM'], device= device)
                    x['LSTM'] = torch.zeros(1, config['input_size']['LSTM'], device= device)

                classical_models['Thompson']['model'] = Thompson(num_arms)
                classical_models['UCB']['model'] = UCB(num_arms, config['c'])
                classical_models['Gittins']['model'] = Gittins(num_arms, gittins_table)

                regrets = {model: [] for model in models}
                
                for t in range(num_trials):
                    optimal = probs.max(dim= -1).values
                    # a single reward outcome for fair evaluation
                    arm_rewards = torch.bernoulli(probs).squeeze(0)

                    # step
                    logits = {model: None for model in trained_models}
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
                        if model in trained_models:
                            pi = torch.distributions.Categorical(logits= logits[model])
                            a[model] = pi.sample()
                            r[model] = arm_rewards[a[model].item()].unsqueeze(0)
                        if model in classical_models:
                            a[model] = models[model]['model'].choice()
                            r[model] = arm_rewards[a[model]].item()
                            models[model]['model'].getReward(a[model], r[model])
                        regrets[model].append(optimal.item() - probs[0, a[model]].item())

                    # update obs
                    if 'DisRNN' in models:
                        x['DisRNN'] = torch.stack([2*a['DisRNN'].float() - 1, 2*r['DisRNN'] - 1], dim= -1)
                    if 'DisLRU' in models:
                        x['DisLRU'] = torch.zeros(1, config['input_size']['DisLRU'], device= device)
                        x['DisLRU'][torch.arange(1, device= device), a['DisLRU']] = 2*r['DisLRU'] - 1
                    if 'LSTM' in models:
                        x['LSTM'] = torch.stack([2*a['LSTM'].float() - 1, 2*r['LSTM'] - 1], dim= -1)

                    # restless bandits
                    if restless:
                        probs += drift * torch.randn(1, num_arms, device= device)
                        probs = torch.clamp(probs, 0, 1)
                        if dependent_arms:
                            probs[:, 1] = 1 - probs[:, 0]

                for model in models:
                    raw_regrets[model].append(np.array(regrets[model]))
                    cumulative_regrets[model].append(np.array(regrets[model]).cumsum())

            return raw_regrets, cumulative_regrets




    # load best models
    for model in trained_models:
        checkpoint = torch.load(checkpoints_path + f'best_{model}.pt')
        models[model]['model'].load_state_dict(checkpoint[f'{model}_state_dict'])
        if 'readout' in models[model]:
            models[model]['readout'].load_state_dict(checkpoint[f'{model}_readout_state_dict'])

    # testing
    raw_regrets, cumulative_regrets = run_tests(10_000)

    # plot cumulative regrets
    plt.figure(figsize= (8,5))
    plt.ylim(0, 4.0)
    for model in models:
        plot_agent(cumulative_regrets[model], models[model]['color'], models[model]['linestyle'], model)
    plt.xlabel('Trial')
    plt.ylabel('Cumulative Regret')
    plt.title('Model Cumulative Regret')
    plt.legend()
    plt.grid()
    plt.savefig(figs_path + 'cumulative_regret.png')
    plt.close()

    # plot optimal arm rates
    plt.figure(figsize= (8,5))
    for model in models:
        plt.plot(optimal_arm_rate(raw_regrets[model]), color= models[model]['color'], linestyle= models[model]['linestyle'], label= model)
    plt.xlabel('Trial')
    plt.ylabel('P(optimal arm chosen)')
    plt.title('Model Optimal Arm Rate')
    plt.legend()
    plt.grid()
    plt.savefig(figs_path + 'optimal_arm_rate.png')
    plt.close()




def main():
    for exp, config in exps.items():
        checkpoints_path = f'checkpoints/{exp}/seed{seed}/'
        figs_path = f'figs/{exp}/seed{seed}/'

        if os.path.exists(checkpoints_path):
            test_res = input(f"Begin testing for experiment: {exp}, seed {seed}? (y/n): ")
            if test_res.lower() == 'n':
                continue

            if os.path.exists(figs_path + 'cumulative_regret.png') or os.path.exists(figs_path  + 'optimal_arm_rate.png'):
                overwrite_res = input(f"There is history for this experiment. Continue? (y/n): ")
                if overwrite_res.lower() == 'n':
                    continue

            print(f"Beginning testing for experiment {exp} bandits, seed {seed}.\n")
            test(config, checkpoints_path, figs_path)


if __name__ == "__main__":
    main()
