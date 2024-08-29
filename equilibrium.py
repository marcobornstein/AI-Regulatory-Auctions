import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def regauc_equilibrium(lambda_max, beta=False):

    num_trials = 1000000
    p_epsilon = 0.75
    diff = np.linspace(0.5, 1.5, 101, endpoint=True)
    i = 0
    pbar = tqdm(total=num_trials)
    actual_utility = np.zeros((num_trials, 2))
    varied_utilities = np.zeros((num_trials, len(diff)))
    while i < num_trials:

        # get two agents values at random
        total_values = np.random.beta(2, 2, size=2) if beta else np.random.uniform(0, 1, size=2)
        lambdas = np.random.uniform(0, lambda_max, size=2)

        """
        # normal all-pay auction
        bids = 0.5*np.square(total_values)
        varied_bid = bids[0] * diff
        varied_utilities[i, :] = - varied_bid
        winner_bool = varied_bid > bids[1]
        varied_utilities[i, winner_bool] += total_values[0]
        """

        # get optimal bids & utility
        v_i_p = total_values * lambdas
        v_i_d = total_values * (1 - lambdas)
        sq_v_i_p = np.square(v_i_p)
        if beta:
            bids = np.minimum(1, p_epsilon + sq_v_i_p * (3 - 8 * v_i_p + 6 * sq_v_i_p))
        else:
            Fv = 2*v_i_p*np.log(1/p_epsilon) / (1-p_epsilon)
            int_Fv = sq_v_i_p*np.log(1/p_epsilon) / (1-p_epsilon)
            bids = np.minimum(1, p_epsilon + v_i_p*Fv - int_Fv)
        utilities = v_i_d - bids + v_i_p * Fv

        # do not count the run if one agent utility is less than participatory
        if np.any(utilities <= 0):
            continue

        # compute actual utility
        winner_idx = np.argmax(bids)
        actual_utility[i, :] = v_i_d - bids
        actual_utility[i, winner_idx] += v_i_p[winner_idx]

        # vary bid effect
        varied_bid = bids[0] * diff
        varied_utilities[i, :] = v_i_d[0] - varied_bid
        winner_bool = varied_bid > bids[1]
        under_pe_bool = varied_bid < p_epsilon
        varied_utilities[i, winner_bool] += v_i_p[0]
        varied_utilities[i, under_pe_bool] = -varied_bid[under_pe_bool]

        i += 1
        pbar.update(1)

    lab = 'beta' if beta else 'uniform'
    mean_varied_utility = np.mean(varied_utilities, axis=0)
    best_idx = np.argmax(mean_varied_utility)
    x_vals = diff*100 - 100

    plt.figure()
    plt.plot(x_vals, mean_varied_utility)
    plt.plot(x_vals[best_idx], mean_varied_utility[best_idx], 'r*', markersize=5, label='Maximizer')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('Percentage Added/Subtracted to Theoretically Optimal Bid (%)')
    plt.ylabel('Average Agent Utility')
    plt.xlim([-50, 50])

    filename = 'regauc_equilibrium_deviation_pe{}_lam{}_{}.png'.format(p_epsilon, lambda_max, lab)
    plt.savefig(filename, dpi=500)
    plt.show()


if __name__ == '__main__':
    lam = 1/2
    regauc_equilibrium(lam, beta=False)
