import numpy as np
import torch
import matplotlib.pyplot as plt
import os, random, copy

from envs import *
from helpers import *

from agents.DisRNN import MyDisRNN
from agents.DisLRU import MyDisLRU

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


colors = {
    'DisRNN': 'blue',
    'DisLRU': 'orange',
    'LSTM': 'green',
    'Thompson': 'gray',
    'UCB': 'lightgray',
    'Gittins': 'black'
}
linestyles = {
    'DisRNN': '-',
    'DisLRU': '-',
    'LSTM': '-',
    'Thompson': '--',
    'UCB': '--',
    'Gittins': '--'
}


exps = {
    'bandits/independent/standard': {
        'name': 'standard independent bandits',
        'env': BanditsEnv(
            config= {
                'D': sample_independent,
                'num_arms': 2,
                'dependent_arms': False,
                'restless': False,
                'drift': 0.0,
                'num_trials': 100
            },
            device= device
        ),
        'models': {'DisLRU'},
        'input_size': {
            'DisRNN': 2,
            'DisLRU': 2,
            'LSTM': 2,
            'Thompson': 2,
            'UCB': 2,
            'Gittins': 2
        },
        'hidden_size': {
            'DisRNN': 5,
            'DisLRU': 5,
            'LSTM': 48
        },
        'output_size': {
            'DisRNN': 2,
            'DisLRU': 2,
            'LSTM': 2
        },
        'gamma': {
            'DisRNN': 0.98,
            'DisLRU': 0.99,
            'LSTM': 0.95,
            'Gittins': 0.99
        },
        'lr': {
            'DisRNN': 5e-4,
            'DisLRU': 5e-3,
            'LSTM': 5e-3
        },
        'batch_size': 32,
        'steps_unrolled': 100,
        'beta_e_annealed': True,
        'beta_e': 0.005,
        'beta_v': 0.05,
        'beta': {
            'DisRNN': {
                'floor': 1e-8,
                'ceil': 1e-4,
                'warmup': {
                    'start': 5000,
                    'end': 10_000
                }
            },
            'DisLRU': {
                'floor': 1e-8,
                'ceil': 1e-6,
                'warmup': {
                    'start': 5000,
                    'end': 10_000
                }
            }
        },
        'train_until_ep': {'LSTM': 200_000},
        'eval_interval': 500,
        'eval_episodes': 1000,
        'search_episodes': 20_000,
        'c': 0.15,
        'color': colors,
        'linestyle': linestyles
    }
}
