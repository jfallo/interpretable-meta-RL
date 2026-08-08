import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os

from agents.DisRNN import MyDisRNN


seed = 40
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# experiments
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

num_arms = 2
exps = {
    'independent': {
        'D': sample_independent,
        'num_trials': 100,
        'restless': False,
        'drift': 0.0,
        'dependent_arms': False,
        'input_size': 2,
        'hidden_size': {
            'DisRNN': 5,
            'LSTM': 48
        },
        'gamma': {
            'DisRNN': 0.98,
            'LSTM': 0.95,
            'gittins': 0.99
        },
        'lr': {
            'DisRNN': 5e-4,
            'LSTM': 5e-3
        },
        'batch_size': 32,
        'steps_unrolled': 100,
        'beta_e_annealed': True,
        'beta_e': 0.005,
        'beta_v': 0.05,
        'beta_floor': 1e-8,
        'beta_ceil': 1e-4,
        'train_LSTM_until_ep': 200_000,
        'eval_interval': 500,
        'eval_episodes': 1000,
        'search_episodes': 50_000,
        'c': 0.15
    },
    'independent/restless': {
        'D': sample_independent,
        'num_trials': 100,
        'restless': True,
        'drift': 0.02,
        'dependent_arms': False,
        'input_size': 2,
        'hidden_size': {
            'DisRNN': 5,
            'LSTM': 48
        },
        'gamma': {
            'DisRNN': 0.98,
            'LSTM': 0.95,
            'gittins': 0.99
        },
        'lr': {
            'DisRNN': 5e-4,
            'LSTM': 5e-3
        },
        'batch_size': 32,
        'steps_unrolled': 100,
        'beta_e_annealed': True,
        'beta_e': 0.005,
        'beta_v': 0.05,
        'beta_floor': 1e-8,
        'beta_ceil': 1e-4,
        'train_LSTM_until_ep': 200_000,
        'eval_interval': 500,
        'eval_episodes': 1000,
        'search_episodes': 50_000,
        'c': 0.15
    },
    'dependent': {
        'D': sample_dependent,
        'num_trials': 100,
        'restless': False,
        'drift': 0.0,
        'dependent_arms': True,
        'input_size': 2,
        'hidden_size': {
            'DisRNN': 5,
            'LSTM': 48
        },
        'gamma': {
            'DisRNN': 0.98,
            'LSTM': 0.95,
            'gittins': 0.98
        },
        'lr': {
            'DisRNN': 5e-4,
            'LSTM': 5e-3
        },
        'batch_size': 32,
        'steps_unrolled': 100,
        'beta_e_annealed': True,
        'beta_e': 0.005,
        'beta_v': 0.05,
        'beta_floor': 1e-8,
        'beta_ceil': 1e-4,
        'train_LSTM_until_ep': 100_000,
        'eval_interval': 500,
        'eval_episodes': 1000,
        'search_episodes': 50_000,
        'c': 0.15
    },
    'dependent/hard': {
        'D': sample_dependent_hard,
        'num_trials': 100,
        'restless': False,
        'drift': 0.0,
        'dependent_arms': True,
        'input_size': 2,
        'hidden_size': {
            'DisRNN': 5,
            'LSTM': 48
        },
        'gamma': {
            'DisRNN': 0.98,
            'LSTM': 0.95,
            'gittins': 0.99
        },
        'lr': {
            'DisRNN': 5e-4,
            'LSTM': 5e-3
        },
        'batch_size': 32,
        'steps_unrolled': 100,
        'beta_e_annealed': True,
        'beta_e': 0.005,
        'beta_v': 0.05,
        'beta_floor': 1e-8,
        'beta_ceil': 1e-4,
        'train_LSTM_until_ep': 100_000,
        'eval_interval': 500,
        'eval_episodes': 1000,
        'search_episodes': 50_000,
        'c': 0.15
    }
}
