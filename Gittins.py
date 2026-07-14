import numpy as np


def _terminal_values(Sigma, n_a, N, gamma, lam):
    k = np.arange(N+1)
    p = (Sigma + k) / (n_a + N)
    
    return np.maximum(p - lam, 0.0) / (1.0 - gamma)


def bmab_value_at_root(Sigma, n, gamma, lam, N):
    V_next = _terminal_values(Sigma, n, N, gamma, lam)

    for s in range(N-1, -1, -1):
        k = np.arange(s+1)
        p = (Sigma + k) / (n + s)
        cont = p * V_next[k+1] + (1-p) * V_next[k]
        V_next = np.maximum(p - lam + gamma * cont, 0.0)

    return V_next[0]


def _find_bracket(value_fn, low, high, expand= 0.05, max_iters= 20):
    i = 0
    while value_fn(low) <= 0 and low > 0 and i < max_iters:
        low = max(0.0, low - expand)
        i += 1
    
    i = 0
    while value_fn(high) > 0 and high < 1 and i < max_iters:
        high = min(1.0, high + expand)
        i += 1
    
    return low, high


def bmab_gittins_index(Sigma, n, gamma, N= 200, tol= 1e-5, lam_bounds= (0.0, 1.0)):
    value_fn = lambda lam : bmab_value_at_root(Sigma, n, gamma, lam, N)

    low, high = _find_bracket(value_fn, *lam_bounds)
    while high - low > tol:
        mid = (low + high) / 2
        if value_fn(mid) > 0:
            low = mid
        else:
            high = mid
    
    return (low + high) / 2


def compute_gittins_table(max_total, gamma, N= 200, tol= 1e-4, warm_start_margin= 0.08):
    size = max_total + 2
    
    table = np.full((size, size), np.nan)
    for total in range(2, size):
        for a in range(1, total):
            b = total - a
            if b < 1 or b >= size:
                continue

            neighbor = None
            if a > 1 and not np.isnan(table[a-1, b]):
                neighbor = table[a-1, b]
            elif b > 1 and not np.isnan(table[a, b-1]):
                neighbor = table[a, b-1]

            if neighbor is not None:
                bounds = (max(0.0, neighbor - warm_start_margin),
                          min(1.0, neighbor + warm_start_margin))
            else:
                bounds = (0.0, 1.0)

            table[a,b] = bmab_gittins_index(a, a+b, gamma, N= N, tol= tol,
                                              lam_bounds= bounds)
    
    return table


class Gittins:
    def __init__(self, num_arms, table):
        self.num_arms = num_arms
        self.table = table
        self.alpha = np.ones(num_arms, dtype= int)
        self.beta = np.ones(num_arms, dtype= int)
    
    def choice(self):
        indices = [self.table[self.alpha[i], self.beta[i]] for i in range(self.num_arms)]
        
        return int(np.argmax(indices))
    
    def getReward(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
