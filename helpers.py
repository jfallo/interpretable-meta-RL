import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from agents.DisRNN import MyDisRNN
from agents.DisLRU import MyDisLRU


def format_matrix(M, name, row_prefix= 'rule', col_prefix= 'dim'):
    M = np.atleast_2d(M)
    n_rows, n_cols = M.shape

    header = len(row_prefix) * ' ' + '     ' + ' '.join(f'{col_prefix}{j:>2}' for j in range(n_cols))
    lines = [f'{name}:', header]
    for i, row in enumerate(M):
        row_str = ' '.join(f'{v:5.2f}' for v in row)
        lines.append(f'{row_prefix}{i:>2} | {row_str}')
    
    return '\n'.join(lines)


# experiment helpers
def sample_independent(batch_size, num_arms, device):
    return torch.rand(batch_size, num_arms, device= device)

def sample_dependent(batch_size, num_arms, device):
    p1 = torch.rand(batch_size, device= device)
    p2 = 1 - p1
    return torch.stack([p1,p2], dim= 1)

def sample_dependent_hard(batch_size, num_arms, device):
    p1 = torch.where(torch.rand(batch_size, device= device) < 0.5,
                     torch.full((batch_size,), 0.4, device= device),
                     torch.full((batch_size,), 0.6, device= device))
    p2 = 1 - p1
    return torch.stack([p1,p2], dim= 1)

def sample_dependent_medium(batch_size, num_arms, device):
    p1 = torch.where(torch.rand(batch_size, device= device) < 0.5,
                     torch.full((batch_size,), 0.25, device= device),
                     torch.full((batch_size,), 0.75, device= device))
    p2 = 1 - p1
    return torch.stack([p1,p2], dim= 1)

def sample_dependent_easy(batch_size, num_arms, device):
    p1 = torch.where(torch.rand(batch_size, device= device) < 0.5,
                     torch.full((batch_size,), 0.1, device= device),
                     torch.full((batch_size,), 0.9, device= device))
    p2 = 1 - p1
    return torch.stack([p1,p2], dim= 1)


def model_constructor(model_name, lr, config, device):
    input_size = config['input_size'][model_name]
    hidden_size = config['hidden_size'][model_name]
    output_size = config['output_size'][model_name]
    training_eps = config['training_eps'][model_name]

    if model_name == 'DisRNN':
        model = MyDisRNN(hidden_size, input_size, output_size).to(device)
        actor_parameters = list(model.parameters())
    elif model_name == 'DisLRU':
        model = MyDisLRU(hidden_size, input_size, output_size).to(device)
        actor_parameters = list(model.parameters())
    elif model_name == 'LSTM':
        model = nn.LSTM(input_size, hidden_size).to(device)
        readout = nn.Linear(hidden_size, output_size).to(device)
        actor_parameters = list(model.parameters()) + list(readout.parameters())
    critic = nn.Sequential(
        nn.Linear(hidden_size + 1, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    ) .to(device)
    critic_parameters = list(critic.parameters())
    parameters = actor_parameters + critic_parameters
    optimizer = torch.optim.Adam(parameters, lr= lr)

    model_dict = {'name': model_name,
                  'input_size': input_size,
                  'hidden_size': hidden_size,
                  'output_size': output_size,
                  'model': model,
                  'critic': critic,
                  'actor_parameters': actor_parameters,
                  'critic_parameters': critic_parameters,
                  'parameters': parameters,
                  'optimizer': optimizer,
                  'training_eps': training_eps}
    if model_name in {'DisRNN', 'DisLRU'}:
        model_dict['bottlenecks'] = ['h', 'x', 'z']
        model_dict['beta'] = config['beta'][model_name]
    if model_name == 'LSTM':
        model_dict['readout'] = readout
    
    return model_dict


# training helpers
def smooth(x, window= 200):
    x = np.asarray(x)
    return x if len(x) < window else np.convolve(x, np.ones(window)/window, mode= 'valid')


def plot_regret_histories(regret_histories, figs_dir, colors):
    plt.figure(figsize= (8,5))
    for model, history in regret_histories.items():
        plt.plot(smooth(history), label= model, color= colors[model])
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.title('Model Training Regret')
    plt.legend()
    plt.grid()
    plt.savefig(figs_dir + 'training_regret.png')
    plt.close()


def print_bottleneck_parameters(model):
    with torch.no_grad():
        print()
        print(model['name'])
        print()
        for bottleneck in model['bottlenecks']:
            if hasattr(model['model'], f'logit_M_{bottleneck}'):
                m = torch.sigmoid(getattr(model['model'], f'logit_M_{bottleneck}'))
            elif hasattr(model['model'], f'M_{bottleneck}'):
                m = getattr(model['model'], f'M_{bottleneck}')
            else:
                continue
            sigma = torch.exp(getattr(model['model'], f'log_sigma_{bottleneck}'))
            
            for name, val in ((f'M_{bottleneck}', m), (f'sigma_{bottleneck}', sigma), (f'M_{bottleneck}/sigma_{bottleneck}', m/sigma)):
                print(format_matrix(val.detach().cpu().numpy(), name, row_prefix= 'lat', col_prefix= 'lat'))
                print()
        print()


def load_checkpoint(checkpoint, models):
    ep = checkpoint['ep']
    regret_histories = checkpoint['regret_histories']
    for model in models:
        models[model]['model'].load_state_dict(checkpoint[f'{model}_state_dict'])
        models[model]['critic'].load_state_dict(checkpoint[f'{model}_critic_state_dict'])
        models[model]['optimizer'].load_state_dict(checkpoint[f'{model}_optimizer_state_dict'])
        if 'readout' in models[model]:
            models[model]['readout'].load_state_dict(checkpoint[f'{model}_readout_state_dict'])

    return ep, regret_histories, models


def build_checkpoint(ep, regret_histories, models):
    checkpoint = {'ep': ep, 
                  'regret_histories': regret_histories}
    for model in models:
        checkpoint[f'{model}_state_dict'] = models[model]['model'].state_dict()
        checkpoint[f'{model}_critic_state_dict'] = models[model]['critic'].state_dict()
        checkpoint[f'{model}_optimizer_state_dict'] = models[model]['optimizer'].state_dict()
        if 'readout' in models[model]:
            checkpoint[f'{model}_readout_state_dict'] = models[model]['readout'].state_dict()
    
    return checkpoint


# testing helpers
def plot_agent(data, T, color, linestyle, label, plot_std= False):
    mean = np.stack(data).mean(axis= 0)
    plt.plot(mean, color= color, linestyle= linestyle, label= label)
    if plot_std:
        std = np.stack(data).std(axis= 0, ddof= 1)
        plt.fill_between(range(T), mean - std, mean + std, alpha= 0.1, color= color, linestyle= linestyle)


def optimal_arm_rate(regrets):
    return (regrets == 0).mean(axis= 0)
