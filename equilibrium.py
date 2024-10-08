import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def regauc_equilibrium(lambda_max, beta=False):

    num_trials = 100000
    p_epsilon = 0.75
    diff = np.linspace(0.5, 1.5, 101, endpoint=True)
    i = 0
    beta_cdf = lambda x: 3 * x ** 2 - 2 * x ** 3
    pbar = tqdm(total=num_trials)
    actual_utility = np.zeros((num_trials, 2))
    varied_utilities = np.zeros((num_trials, len(diff)))
    while i < num_trials:

        # get two agents values at random
        total_values = np.random.beta(2, 2, size=2) if beta else np.random.uniform(0, 1, size=2)
        lambdas = np.random.uniform(0, lambda_max, size=2)

        # get optimal bids & utility
        v_i_p = total_values * lambdas
        v_i_d = total_values * (1 - lambdas)
        Fv = np.empty_like(v_i_p)
        int_Fv = np.empty_like(v_i_p)
        lower_bool = v_i_p <= p_epsilon / 2
        if beta:
            den = 1 - beta_cdf(p_epsilon)
            Fv[lower_bool] = 6 * v_i_p[lower_bool] * (p_epsilon ** 2 - 2 * p_epsilon + 1) / den
            Fv[~lower_bool] = (8 * v_i_p[~lower_bool] ** 3 - 12 * v_i_p[~lower_bool] ** 2 + 6 * v_i_p[
                ~lower_bool] + 2 * p_epsilon ** 3 - 3 * p_epsilon ** 2) / den
            int_Fv[lower_bool] = 3 * v_i_p[lower_bool] ** 2 * (p_epsilon ** 2 - 2 * p_epsilon + 1) / den
            int_Fv[~lower_bool] = (2 * v_i_p[~lower_bool] ** 4 - 4 * v_i_p[~lower_bool] ** 3 + 3 * v_i_p[
                ~lower_bool] ** 2 + v_i_p[~lower_bool] * (
                                               2 * p_epsilon ** 3 - 3 * p_epsilon ** 2) + 0.5 * p_epsilon ** 3 - (
                                               3 / 8) * p_epsilon ** 4) / den
        else:
            Fv[lower_bool] = 2 * v_i_p[lower_bool] * np.log(p_epsilon) / (p_epsilon-1)
            Fv[~lower_bool] = (2 * v_i_p[~lower_bool] * (np.log(2*v_i_p[~lower_bool]) - 1) + p_epsilon) / (p_epsilon - 1)
            int_Fv[lower_bool] = np.square(v_i_p[lower_bool])*np.log(p_epsilon) / (p_epsilon-1)
            int_Fv[~lower_bool] = (4*np.square(v_i_p[~lower_bool]) * (2*np.log(2*v_i_p[~lower_bool]) - 3) + 8*p_epsilon*v_i_p[~lower_bool] - p_epsilon**2) / (8*(p_epsilon - 1))
        bids = np.minimum(1, p_epsilon + v_i_p * Fv - int_Fv)
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
    x_vals = diff*100 - 100

    plt.figure()
    plt.axvline(x=0, ymin=0, ymax=1, color='r', linestyle=':', label='Optimal Bid')
    plt.plot(x_vals, mean_varied_utility)
    plt.legend(loc='best', fontsize="15")
    plt.grid()
    plt.xlabel('Percent Deviated from Optimal Bid (%)', fontsize="15", fontweight='bold')
    plt.ylabel('Average Agent Utility', fontsize="15", fontweight='bold')
    plt.xlim([-50, 50])

    filename = 'regauc_equilibrium_deviation_pe{}_lam{}_{}.png'.format(p_epsilon, lambda_max, lab)
    plt.savefig(filename, dpi=500)
    plt.show()


if __name__ == '__main__':
    lam = 1/2
    regauc_equilibrium(lam, beta=True)
