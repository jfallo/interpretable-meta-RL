import numpy as np
import torch
import matplotlib.pyplot as plt


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


# training helpers
def smooth(x, window= 200):
    return np.convolve(x, np.ones(window)/window, mode= 'valid')


def plot_regret_histories(regret_histories, figs_dir, colors):
    phase1 = {model: smooth(np.array(regret_histories['phase1'][model])) for model in regret_histories['phase1']}
    plt.figure(figsize= (8,5))
    for model, history in phase1.items():
        plt.plot(history, label= model, color= colors[model])
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.title('Phase 1 Model Regret')
    plt.legend()
    plt.grid()
    plt.savefig(figs_dir + 'training_regret_phase1.png')
    plt.close()

    phase2 = {model: smooth(np.array(regret_histories['phase2'][model])) for model in regret_histories['phase2']}
    if phase2:
        plt.figure(figsize= (8,5))
        for model, history in phase2.items():
            plt.plot(history, label= model, color= colors[model])
        plt.xlabel('Episode')
        plt.ylabel('Regret')
        plt.title('Phase 2 Model Regret')
        plt.legend()
        plt.grid()
        plt.savefig(figs_dir + 'training_regret_phase2.png')
        plt.close()


def print_bottleneck_parameters(model):
    with torch.no_grad():
        print()
        print(model['name'])
        print()
        for bottleneck in model['bottlenecks']:
            m = torch.sigmoid(getattr(model['model'], f'logit_M_{bottleneck}'))
            print(format_matrix(m, f'M_{bottleneck}', row_prefix= 'lat', col_prefix= 'lat'))
            print()
        print()


def disentangled(model, low= 0.1, high= 0.9):
    with torch.no_grad():
        checks = []
        for bottleneck in model['bottlenecks']:
            m = torch.sigmoid(getattr(model['model'], f'logit_M_{bottleneck}'))
            sigma = torch.exp(getattr(model['model'], f'log_sigma_{bottleneck}'))
            checks.append(((m <= low) | (m >= high)).all() and ((sigma <= low) | (sigma >= high)).all())

        return bool(all(checks))


def bottlenecks_converged(model, prev_state_dict, tol= 0.02):
    with torch.no_grad():
        checks = []
        for bottleneck in model['bottlenecks']:
            prev_m = torch.sigmoid(prev_state_dict[f'logit_M_{bottleneck}'])
            prev_sigma = torch.exp(prev_state_dict[f'log_sigma_{bottleneck}'])
            m = torch.sigmoid(getattr(model['model'], f'logit_M_{bottleneck}'))
            sigma = torch.exp(getattr(model['model'], f'log_sigma_{bottleneck}'))
            checks.append((torch.abs(m - prev_m) < tol).all() and (torch.abs(sigma - prev_sigma) < tol).all())

        return bool(all(checks))


def load_checkpoint(checkpoint, models):
    ep = checkpoint['ep']
    prev_state_dicts = checkpoint['prev_state_dicts']
    training_phase = checkpoint['training_phase']
    regret_histories = checkpoint['regret_histories']
    for model in models:
        models[model]['model'].load_state_dict(checkpoint[f'{model}_state_dict'])
        models[model]['critic'].load_state_dict(checkpoint[f'{model}_critic_state_dict'])
        models[model]['optimizer'].load_state_dict(checkpoint[f'{model}_optimizer_state_dict'])
        if 'readout' in models[model]:
            models[model]['readout'].load_state_dict(checkpoint[f'{model}_readout_state_dict'])
        if 'bottlenecks' in models[model]:
            models[model]['converged'] = checkpoint[f'{model}_converged']

    return ep, prev_state_dicts, training_phase, regret_histories, models


def build_checkpoint(ep, prev_state_dicts, training_phase, regret_histories, models):
    checkpoint = {
        'ep': ep, 
        'prev_state_dicts': prev_state_dicts,
        'training_phase': training_phase,
        'regret_histories': regret_histories
    }
    for model in models:
        checkpoint[f'{model}_state_dict'] = models[model]['model'].state_dict()
        checkpoint[f'{model}_critic_state_dict'] = models[model]['critic'].state_dict()
        checkpoint[f'{model}_optimizer_state_dict'] = models[model]['optimizer'].state_dict()
        if 'readout' in models[model]:
            checkpoint[f'{model}_readout_state_dict'] = models[model]['readout'].state_dict()
        if 'bottlenecks' in models[model]:
            checkpoint[f'{model}_converged'] = models[model]['converged']
    
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
