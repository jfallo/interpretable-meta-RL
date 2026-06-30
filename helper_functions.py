import numpy as np


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
