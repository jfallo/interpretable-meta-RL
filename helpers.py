import numpy as np
import torch


def format_matrix(M, name, row_prefix= 'rule', col_prefix= 'dim'):
    M = np.atleast_2d(M)
    n_rows, n_cols = M.shape

    header = '      ' + ' '.join(f'{col_prefix}{j:>2}' for j in range(n_cols))
    lines = [f'{name}:', header]
    for i, row in enumerate(M):
        row_str = ' '.join(f'{v:5.2f}' for v in row)
        lines.append(f'{row_prefix}{i:>2} | {row_str}')
    
    return '\n'.join(lines)


def smooth(x, window= 200):
    return np.convolve(x, np.ones(window)/window, mode= 'valid')


def print_bottleneck_parameters(DisRNN):
    M_h = torch.sigmoid(DisRNN.logit_M_h).detach().cpu().numpy()
    sigma_h = torch.exp(DisRNN.log_sigma_h).detach().cpu().numpy()

    M_x = torch.sigmoid(DisRNN.logit_M_x).detach().cpu().numpy()
    sigma_x = torch.exp(DisRNN.log_sigma_x).detach().cpu().numpy()

    M_z = torch.sigmoid(DisRNN.logit_M_z).detach().cpu().numpy()
    sigma_z = torch.exp(DisRNN.log_sigma_z).detach().cpu().numpy()

    print()
    print(format_matrix(M_h, 'M_h', row_prefix= 'rule', col_prefix= 'lat'))
    print()
    print(format_matrix(sigma_h, 'sigma_h', row_prefix= 'rule', col_prefix= 'lat'))
    print()
    print(format_matrix(M_x, 'M_x', row_prefix= 'rule', col_prefix= 'obs'))
    print()
    print(format_matrix(sigma_x, 'sigma_x', row_prefix= 'rule', col_prefix= 'lat'))
    print()
    print(format_matrix(M_z.reshape(1,-1), 'M_z', row_prefix= 'lat', col_prefix= 'lat'))
    print()
    print(format_matrix(sigma_z, 'sigma_z', row_prefix= 'rule', col_prefix= 'lat'))   
    print()
    print()
