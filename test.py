from config import *

from agents.Thompson import Thompson
from agents.UCB import UCB
from agents.Gittins import compute_gittins_table, Gittins

num_tests = 10_000


def evaluate(config, checkpoints_path, figs_path):
    # build env
    env = config['env']['env'](config= config['env']['config'], 
                               device= config['env']['device'])

    # build Gittins index table
    gittins_table = compute_gittins_table(max_total= env.T+1, gamma= config['gamma']['Gittins'], N= 200, tol= 1e-4)
    
    # initialize models
    colors = config['color']
    linestyles = config['linestyle']

    trained_models = config['models']
    classical_models = {'Thompson', 'UCB', 'Gittins'}
    all_models = set(trained_models) | classical_models

    models = {model: model_constructor(model, config, device)
              for model in trained_models}
    models['Thompson'] = {'model': Thompson(config['input_size']['Thompson'])}
    models['UCB'] = {'model': UCB(config['input_size']['UCB'], config['c'])}
    models['Gittins'] = {'model': Gittins(config['input_size']['Gittins'], gittins_table)}

    env.build_models(models)


    # --- testing ---
    
    with torch.no_grad():
        for model in trained_models:
            checkpoint = torch.load(checkpoints_path + f'best_{model}.pt', map_location= device)
            models[model]['model'].load_state_dict(checkpoint[f'{model}_state_dict'])
            if 'readout' in models[model]:
                models[model]['readout'].load_state_dict(checkpoint[f'{model}_readout_state_dict'])

        raw_regrets = {model: [] for model in all_models}
        cumulative_regrets = {model: [] for model in all_models}
        for _ in range(num_tests):
            for model in trained_models:
                models[model]['model'].eval()          
            env.reset(all_models, testing= True)

            regrets = env.eval_episode(all_models)
            for model in all_models:
                r = regrets[model].detach().cpu().numpy()
                raw_regrets[model].append(r)
                cumulative_regrets[model].append(r.cumsum())


    # plot cumulative regrets
    plt.figure(figsize= (8,5))
    plt.ylim(0, 4.0)
    for model in models:
        plot_agent(cumulative_regrets[model], env.T, colors[model], linestyles[model], model)
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
        plt.plot(optimal_arm_rate(np.stack(raw_regrets[model])), color= colors[model], linestyle= linestyles[model], label= model)
    plt.xlabel('Trial')
    plt.ylabel('P(optimal arm chosen)')
    plt.title('Model Optimal Arm Rate')
    plt.legend()
    plt.grid()
    plt.savefig(figs_path + 'optimal_arm_rate.png')
    plt.close()
