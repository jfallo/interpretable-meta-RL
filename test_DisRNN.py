import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from generate_data import generate_Q_learning_data, generate_actor_critic_data, generate_batch
from DisRNN import DisRNN

#DATA_GENERATOR = generate_actor_critic_data
DATA_GENERATOR = generate_Q_learning_data


def format_matrix(M, name, row_prefix= 'rule', col_prefix= 'dim'):
    M = np.atleast_2d(M)
    n_rows, n_cols = M.shape

    header = '      ' + ' '.join(f'{col_prefix}{j:>2}' for j in range(n_cols))
    lines = [f'{name}:', header]
    for i, row in enumerate(M):
        row_str = ' '.join(f'{v:5.2f}' for v in row)
        lines.append(f'{row_prefix}{i:>2} | {row_str}')
    
    return '\n'.join(lines)


# example data for model structure
history, q = DATA_GENERATOR()
X = torch.tensor(
    [[2*h['a'] - 1, 2*h['r'] - 1] for h in history],
    dtype= torch.float32
).unsqueeze(1)
Y = torch.tensor(
    [h['a'] for h in history],
    dtype= torch.long
)


# initialize DisRNN model and Adam optimizer
model = DisRNN(m= 5, n= X.shape[-1], q= q.shape[0])
optimizer = torch.optim.Adam(model.parameters(), lr= 5e-3)


# training hyperparameters
steps = 30_000
batch_size = 32
beta_max = 1e-3
beta_warmup_start = 1000
beta_warmup_end = 4000


# training
losses = []
losses_softmax = []
losses_bottleneck = []

for step in range(steps):
    # update beta
    beta = beta_max * min(step - beta_warmup_start / beta_warmup_start - beta_warmup_start, 1.0)
    beta = 0 if step < beta_warmup_start else beta

    # put the model in training mode
    model.train()
    # clear gradients from previous step
    optimizer.zero_grad()

    # generate a new sequence
    # want to learn general structure instead of a single sequence
    X, Y = generate_batch(batch_size, DATA_GENERATOR)

    # get DisRNN outputs
    logits, latents, bottleneck_losses = model(X)

    # calculate loss
    loss_softmax = F.cross_entropy(logits[:-1].reshape(-1, model.q), Y[1:].reshape(-1))
    loss_bottleneck = sum(loss.mean() for loss in bottleneck_losses.values())
    loss = loss_softmax + beta * loss_bottleneck

    losses.append(loss.item())
    losses_softmax.append(loss_softmax.item())
    losses_bottleneck.append(loss_bottleneck.item())

    # compute gradients via backpropagation
    loss.backward()
    # update parameters to minimize loss
    optimizer.step()

    if step % 250 == 0:
        M_h = torch.sigmoid(model.logit_M_h).detach().numpy()
        M_x = torch.sigmoid(model.logit_M_x).detach().numpy()
        M_z = torch.sigmoid(model.logit_M_z).detach().numpy()

        print(
            f'step {step:6d} | loss {loss.item():.4f} | CE {loss_softmax.item():.4f} | '
            f'KL {loss_bottleneck.item():.4f} | beta {beta:.4f}'
        )
        print(format_matrix(M_h, 'M_h', row_prefix= 'rule', col_prefix= 'lat'))
        print(format_matrix(M_x, 'M_x', row_prefix= 'rule', col_prefix= 'obs'))
        print(format_matrix(M_z.reshape(1, -1), 'M_z', row_prefix= 'lat', col_prefix= 'lat'))
        print()


# plot loss
fig, axes = plt.subplots(3, 1, figsize= (10,8), sharex= True)

axes[0].plot(losses)
axes[0].set_ylabel('total loss')

axes[1].plot(losses_softmax)
axes[1].set_ylabel('cross entropy loss')

axes[2].plot(losses_bottleneck)
axes[2].set_ylabel('bottleneck loss')

axes[2].set_xlabel('step')

plt.tight_layout()
plt.show()
