import torch
import numpy as np
import argparse, random, os

from config import exps
from train import train


torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type= int, default= 40)
parser.add_argument('--models', nargs= '+', default= None)
parser.add_argument('--lr', type= float, default= None)
parser.add_argument('--gamma', type= float, default= None)
parser.add_argument('--training_eps', type= int, default= None)
parser.add_argument('--resume', default= None)
args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# experiment setup
exp = 'bandits/independent/standard'
config = exps[exp]
device = config['env']['device']
if args.models:
    config['models'] = args.models
if args.lr:
    for model in config['models']:
        config['lr'][model] = args.lr
if args.gamma:
    for model in config['models']:
        config['gamma'][model] = args.gamma
if args.training_eps:
    for model in config['models']:
        config['training_eps'][model] = args.training_eps

tag = os.path.join('_'.join(config['models']), f'lr{args.lr}_gamma{args.gamma}')
checkpoints_dir = f'checkpoints/{exp}/{tag}/seed{seed}/'
figs_dir = f'figs/{exp}/{tag}/seed{seed}/'
os.makedirs(checkpoints_dir, exist_ok= True)
os.makedirs(figs_dir, exist_ok= True)


# run training
print(f'Device: {device}')
print(f"Beginning training for {config['name']} experiment, seed {seed}, models {config['models']}.")
train(config, figs_dir, checkpoints_dir, checkpoint_path= args.resume)
