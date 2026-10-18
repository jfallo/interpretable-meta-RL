from config import *

from train import train
from test import test


seed = 40
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# hyperparameter grid points
lrs = [5e-4, 1e-4, 5e-3, 1e-3]
gammas = [0.9, 0.933, 0.95, 0.98, 0.99]


# experiment setup
exp = 'bandits/independent/standard'
config = exps[exp]

checkpoints_dir = f'checkpoints/{exp}/seed{seed}/'
os.makedirs(checkpoints_dir, exist_ok= True)
figs_dir = f'figs/{exp}/seed{seed}/'
os.makedirs(figs_dir, exist_ok= True)


# run training
print(f"Beginning training for {config['name']} experiment, seed {seed}.")
train(config, figs_dir, checkpoints_dir)
