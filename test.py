from config import *

from agents.Thompson import Thompson
from agents.UCB import UCB
from agents.Gittins import compute_gittins_table, Gittins


def test(config, checkpoints_path, figs_path):
    # set task
    D = config['D']
    num_trials = config['num_trials']
    restless = config['restless']
    drift = config['drift']
    dependent_arms = config['dependent_arms']


    # initialize models
    input_size = config['input_size']

    DisRNN_hidden_size = config['hidden_size']['DisRNN']
    DisRNN = MyDisRNN(DisRNN_hidden_size, input_size, num_arms).to(device)

    LSTM_hidden_size = config['hidden_size']['LSTM']
    LSTM = torch.nn.LSTM(input_size, LSTM_hidden_size).to(device)
    LSTM_readout = torch.nn.Linear(LSTM_hidden_size, num_arms).to(device)


    # testing helpers
    def plot_agent(data, color, linestyle, label, plot_std= False):
        mean = np.stack(data).mean(axis= 0)
        plt.plot(mean, color= color, linestyle= linestyle, label= label)
        
        if plot_std:
            std = np.stack(data).std(axis= 0, ddof= 1)
            plt.fill_between(range(num_trials), mean - std, mean + std, alpha= 0.1, color= color, linestyle= linestyle)


    def run_tests(num_tests):
        DisRNN_raw_regrets = []
        LSTM_raw_regrets = []
        thompson_raw_regrets = []
        ucb_raw_regrets = []
        gittins_raw_regrets = []

        DisRNN_cumulative_regrets = []
        LSTM_cumulative_regrets = []
        thompson_cumulative_regrets = []
        ucb_cumulative_regrets = []
        gittins_cumulative_regrets = []
        for _ in range(num_tests):
            probs = D(1, num_arms, device)
            
            # reset DisRNN state
            DisRNN.eval()
            DisRNN_h = torch.zeros(1, DisRNN_hidden_size, device= device)
            DisRNN_x = torch.zeros(1, input_size, device= device)

            # reset LSTM state
            LSTM.eval()
            LSTM_h = torch.zeros(1, 1, LSTM_hidden_size, device= device)
            LSTM_c = torch.zeros(1, 1, LSTM_hidden_size, device= device)
            LSTM_x = torch.zeros(1, input_size, device= device)

            # test models
            thompson = Thompson(num_arms)
            ucb = UCB(num_arms, c)
            gittins = Gittins(num_arms, gittins_table)


            DisRNN_regrets = []
            LSTM_regrets = []
            thompson_regrets = []
            ucb_regrets = []
            gittins_regrets = []
            with torch.no_grad():
                for t in range(num_trials):
                    optimal = probs.max(dim= -1).values

                    # a single reward outcome for all agents for fair evaluation
                    arm_rewards = torch.bernoulli(probs).squeeze(0)

                    # DisRNN step
                    DisRNN_h, _ = DisRNN.step(DisRNN_h, DisRNN_x)
                    DisRNN_logits = DisRNN.out(DisRNN_h)

                    DisRNN_pi = torch.distributions.Categorical(logits= DisRNN_logits)
                    DisRNN_a = DisRNN_pi.sample()
                    DisRNN_r = arm_rewards[DisRNN_a.item()].unsqueeze(0)
                    DisRNN_x = torch.stack([2*DisRNN_a.float() - 1, 2*DisRNN_r - 1], dim= -1)
                    DisRNN_regrets.append((optimal - probs[0, DisRNN_a]).cpu())

                    # LSTM step
                    LSTM_out, (LSTM_h, LSTM_c) = LSTM(LSTM_x.unsqueeze(0), (LSTM_h, LSTM_c))
                    LSTM_logits = LSTM_readout(LSTM_out.squeeze(0))

                    LSTM_pi = torch.distributions.Categorical(logits= LSTM_logits)
                    LSTM_a = LSTM_pi.sample()
                    LSTM_r = arm_rewards[LSTM_a.item()].unsqueeze(0)
                    LSTM_x = torch.stack([2*LSTM_a.float() - 1, 2*LSTM_r - 1], dim= -1)
                    LSTM_regrets.append((optimal - probs[0, LSTM_a]).cpu())

                    # Thompson step
                    thompson_a = thompson.choice()
                    thompson_r = arm_rewards[thompson_a].item()
                    thompson.getReward(thompson_a, thompson_r)
                    thompson_regrets.append(optimal.item() - probs[0, thompson_a].item())

                    # UCB step
                    ucb_a = ucb.choice()
                    ucb_r = arm_rewards[ucb_a].item()
                    ucb.getReward(ucb_a, ucb_r)
                    ucb_regrets.append(optimal.item() - probs[0, ucb_a].item())

                    # Gittins step
                    gittins_a = gittins.choice()
                    gittins_r = arm_rewards[gittins_a].item()
                    gittins.getReward(gittins_a, gittins_r)
                    gittins_regrets.append(optimal.item() - probs[0, gittins_a].item())


                    # restless bandits
                    if restless:
                        probs += drift * torch.randn(1, num_arms, device= device)
                        probs = torch.clamp(probs, 0, 1)
                        if dependent_arms:
                            probs[:, 1] = 1 - probs[:, 0]

                    
            DisRNN_raw_regrets.append(np.array(DisRNN_regrets))
            LSTM_raw_regrets.append(np.array(LSTM_regrets))
            thompson_raw_regrets.append(np.array(thompson_regrets))
            ucb_raw_regrets.append(np.array(ucb_regrets))
            gittins_raw_regrets.append(np.array(gittins_regrets))

            DisRNN_cumulative_regrets.append(np.array(DisRNN_regrets).cumsum())
            LSTM_cumulative_regrets.append(np.array(LSTM_regrets).cumsum())
            thompson_cumulative_regrets.append(np.array(thompson_regrets).cumsum())
            ucb_cumulative_regrets.append(np.array(ucb_regrets).cumsum())
            gittins_cumulative_regrets.append(np.array(gittins_regrets).cumsum())


        return {
            'DisRNN': {
                'raw_regrets': DisRNN_raw_regrets,
                'cumulative_regrets': DisRNN_cumulative_regrets,
            },
            'LSTM': {
                'raw_regrets': LSTM_raw_regrets,
                'cumulative_regrets': LSTM_cumulative_regrets,
            },
            'Thompson': {
                'raw_regrets': thompson_raw_regrets,
                'cumulative_regrets': thompson_cumulative_regrets,
            },
            'UCB': {
                'raw_regrets': ucb_raw_regrets,
                'cumulative_regrets': ucb_cumulative_regrets,
            },
            'Gittins': {
                'raw_regrets': gittins_raw_regrets,
                'cumulative_regrets': gittins_cumulative_regrets,
            },
        }

            
    def optimal_arm_rate(raw_regrets):
        regrets = np.stack(raw_regrets)

        return (regrets == 0).mean(axis= 0)




    # build Gittins index table
    gittins_table = compute_gittins_table(max_total= num_trials+1, gamma= 0.99, N= 200, tol= 1e-4)

    # load best models
    best_DisRNN = torch.load(checkpoints_path + 'best_DisRNN.pt')
    DisRNN.load_state_dict(best_DisRNN['DisRNN_state_dict'])

    best_LSTM = torch.load(checkpoints_path + 'best_LSTM.pt')
    LSTM.load_state_dict(best_LSTM['LSTM_state_dict'])
    LSTM_readout.load_state_dict(best_LSTM['LSTM_readout_state_dict'])

    # testing
    results = run_tests(1000)

    # plot cumulative regrets
    plt.figure(figsize= (8,5))
    plt.ylim(0, 4.0)
    plot_agent(results['DisRNN']['cumulative_regrets'], 'blue', '-', 'DisRNN')
    plot_agent(results['LSTM']['cumulative_regrets'], 'green', '-', 'LSTM')
    plot_agent(results['Thompson']['cumulative_regrets'], 'gray', '--', 'Thompson')
    plot_agent(results['UCB']['cumulative_regrets'], 'lightgray', '--', 'UCB')
    plot_agent(results['Gittins']['cumulative_regrets'], 'black', '--', 'Gittins')
    plt.xlabel('Trial')
    plt.ylabel('Cumulative Regret')
    plt.title('Model Cumulative Regret')
    plt.legend()
    plt.grid()
    plt.savefig(figs_path + 'cumulative_regret.png')
    plt.close()

    # plot optimal arm rates
    plt.figure(figsize= (8,5))
    plt.plot(optimal_arm_rate(results['DisRNN']['raw_regrets']), color= 'blue', linestyle= '-', label= 'DisRNN')
    plt.plot(optimal_arm_rate(results['LSTM']['raw_regrets']), color= 'green', linestyle= '-', label= 'LSTM')
    plt.plot(optimal_arm_rate(results['Thompson']['raw_regrets']), color= 'gray', linestyle= '--', label= 'Thompson')
    plt.plot(optimal_arm_rate(results['UCB']['raw_regrets']), color= 'lightgray', linestyle= '--', label= 'UCB')
    plt.plot(optimal_arm_rate(results['Gittins']['raw_regrets']), color= 'black', linestyle= '--', label= 'Gittins')
    plt.xlabel('Trial')
    plt.ylabel('P(optimal arm chosen)')
    plt.title('Model Optimal Arm Rate')
    plt.legend()
    plt.grid()
    plt.savefig(figs_path + 'optimal_arm_rate.png')
    plt.close()




def main():
    for exp, config in exps.items():
        begin_testing = True
        checkpoints_path = f'checkpoints/{exp}/seed{seed}/'
        figs_path = f'figs/{exp}/seed{seed}/'
        if os.path.exists(figs_path + 'cumulative_regret.png') or os.path.exists(figs_path  + 'optimal_arm_rate.png'):
            res = input(
                f"There is history for experiment: {exp} bandits, seed: {seed}. "
                "Do you want to overwrite it? (y/n): "
            )
            begin_testing = res.lower() == 'y'

        if begin_testing:
            print(f"Beginning testing for experiment {exp} bandits, seed {seed}.\n")
            test(config, checkpoints_path, figs_path)


if __name__ == "__main__":
    main()
