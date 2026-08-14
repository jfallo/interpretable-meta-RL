from config import *

from train import train
from test import test


for exp, config in exps.items():
    exp_name = config['name']

    run_exp = input(f"Do you want to run the {exp_name} experiment? (y/n): ")
    if run_exp.lower() == 'n':
        continue

    while True:
        try:
            seed = int(input("Enter a seed for the experiment: "))
            break
        except ValueError:
            print("Please enter a valid integer seed.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    checkpoints_dir = f'checkpoints/{exp}/seed{seed}/'
    figs_dir = f'figs/{exp}/seed{seed}/'

    checkpoint_path = ''
    resume_training = False
    skip_training = False
    if os.path.exists(checkpoints_dir) or os.path.exists(figs_dir):
        cont = input("There is history for this experiment. Continue? (y/n): ")
        if cont.lower() == 'n':
            continue

        run_training = input("Do you want to run training? (y/n): ")
        skip_training = run_training.lower() == 'n'

        if not skip_training:
            resume_checkpoint = input("Resume a checkpoint? (y/n): ")
            resume_training = resume_checkpoint.lower() == 'y'

            if resume_training:
                while True:
                    checkpoint_path = input("Checkpoint path: ")
                    if os.path.exists(checkpoint_path):
                        break

                    try_again = input(
                        f"Cannot find checkpoint: {checkpoint_path} \n"
                        "Enter another checkpoint? (y/n): "
                    )
                    if try_again.lower() == 'n':
                        resume_training = False
                        checkpoint_path = ''
                        break

    os.makedirs(checkpoints_dir, exist_ok= True)
    os.makedirs(figs_dir, exist_ok= True)

    if not skip_training:
        if resume_training:
            print(f"Resuming training for the {exp_name} experiment, seed {seed} from {checkpoint_path}.")
        else:
            print(f"Beginning training for the {exp_name} experiment, seed {seed}.")

        train(config, checkpoint_path, checkpoints_dir, figs_dir)
        print("Training complete.")

    run_testing = input("Do you want to run testing? (y/n): ")
    if run_testing.lower() == 'n':
        continue

    if os.path.exists(figs_dir + 'cumulative_regret.png') or os.path.exists(figs_dir + 'optimal_arm_rate.png'):
        overwrite = input("There is testing history for this experiment. Overwrite it? (y/n): ")
        if overwrite.lower() == 'n':
            continue

    print(f"Beginning testing for the {exp_name} experiment, seed {seed}.")

    test(config, checkpoints_dir, figs_dir)