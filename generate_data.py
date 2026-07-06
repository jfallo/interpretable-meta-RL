import torch


def get_reward(a, probs):
    return 1 if torch.rand(()) < probs[a] else 0


def generate_Q_learning_data(T= 500, alpha= 0.3, beta= 3, drift= 0.02):
    probs = torch.rand(2)
    
    q = 0.5 * torch.ones(2)
    
    history = []
    for t in range(T):
        p = torch.sigmoid(beta * (q[0] - q[1]))
        
        a = 0 if torch.rand(()) < p else 1
        r = get_reward(a, probs)

        q[a] = (1 - alpha) * q[a] + alpha * r

        probs += drift * torch.randn(2)
        probs = torch.clamp(probs, 0, 1)

        history.append({
            't': t,
            'a': a,
            'r': r,
            'q': q.clone()
        })

    return history, q


def generate_actor_critic_data(T= 500, alpha= 0.8, lr= 0.8, fr= 1e-4, drift= 0.02):
    probs = torch.rand(2)

    V = torch.tensor(0.0)
    Theta = torch.tensor(0.0)
    
    history = []
    for t in range(T):
        p = torch.sigmoid(Theta)
        
        a = 0 if torch.rand(()) < p else 1
        r = get_reward(a, probs)
        pi = torch.tensor([p, 1-p])

        sign = 1 if a == 0 else -1
        Theta = (1 - fr) * Theta + sign * 2 * lr * (r - V) * pi[1-a]
        V = (1 - alpha) * V + alpha * r

        probs += drift * torch.randn(2)
        probs = torch.clamp(probs, 0, 1)

        history.append({
            't': t,
            'a': a,
            'r': r,
            'V': V,
            'Theta': Theta
        })

    return history


def generate_batch(B, data_generator, **kwargs):
    histories = []
    for _ in range(B):
        history, _ = data_generator(**kwargs)
        histories.append(history)

    X = torch.tensor(
        [[[2*h['a'] - 1, 2*h['r'] - 1] for h in history] for history in histories],
        dtype= torch.float32
    ).permute(1, 0, 2)
    Y = torch.tensor(
        [[h['a'] for h in history] for history in histories],
        dtype= torch.long
    ).permute(1, 0)

    return X, Y
