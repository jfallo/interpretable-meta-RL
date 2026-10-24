import torch

from envs import BanditsEnv
from helpers import sample_independent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


colors = {'DisRNN': 'blue',
          'DisLRU': 'orange',
          'LSTM': 'green',
          'Thompson': 'gray',
          'UCB': 'lightgray',
          'Gittins': 'black'}

linestyles = {'DisRNN': '-',
              'DisLRU': '-',
              'LSTM': '-',
              'Thompson': '--',
              'UCB': '--',
              'Gittins': '--'}


exps = {
    'bandits/independent/standard': {
        'name': 'standard independent bandits',
        'env': {
            'env': BanditsEnv,
            'config': {'D': sample_independent,
                       'num_arms': 2,
                       'dependent_arms': False,
                       'restless': False,
                       'drift': 0.0,
                       'num_trials': 100},
            'device': device
        },
        'models': ['DisRNN', 'DisLRU', 'LSTM'],
        'input_size': {
            'DisRNN': 3,
            'DisLRU': 3,
            'LSTM': 3,
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
            'DisLRU': 0.98,
            'LSTM': 0.98,
            'Gittins': 0.99
        },
        'lr': {
            'DisRNN': 5e-4,
            'DisLRU': 5e-4,
            'LSTM': 5e-3
        },
        'batch_size': 256,
        'steps_unrolled': 100,
        'beta_e': {
            'floor': 0.005,
            'ceil': 0.1,
            'anneal': {'start': 0,
                       'end': 5_000},
            'tau': 1_000
        },
        'beta_v': 0.05,
        'beta': {
            'DisRNN': {
                'ceil': 1e-4,
                'warmup': {'start': 10_000,
                           'end': 20_000}
            },
            'DisLRU': {
                'ceil': 1e-4,
                'warmup': {'start': 10_000,
                           'end': 20_000}
            }
        },
        'training_eps': {
            'DisRNN': 400_000,
            'DisLRU': 400_000,
            'LSTM': 200_000,
        },
        'c': 0.15,
        'color': colors,
        'linestyle': linestyles
    }
}
