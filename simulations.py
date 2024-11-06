import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def regauc_sim(lambda_max, beta=False):

    num_agents = 100000
    beta_cdf = lambda x: 3 * x ** 2 - 2 * x ** 3
    agent_total_value = np.random.beta(2, 2, size=num_agents) if beta else np.random.uniform(
        0, 1, size=num_agents)
    agent_total_lambda = np.random.uniform(0, lambda_max, size=num_agents)

    v_i_p = agent_total_value * agent_total_lambda
    v_i_d = agent_total_value * (1 - agent_total_lambda)

    epsilons = np.linspace(0.01, 1, 99, endpoint=False)
    ra_rates = np.empty_like(epsilons)
    ra_avgerage_bids = np.empty_like(epsilons)
    st_rates = np.empty_like(epsilons)
    st_average_bids = np.empty_like(epsilons)
    Fv = np.empty_like(v_i_p)
    int_Fv = np.empty_like(v_i_p)
    i = 0
    for p_epsilon in tqdm(epsilons):
        lower_bool = v_i_p <= p_epsilon / 2
        # reg auc participation rate
        if beta:
            den = 1 - beta_cdf(p_epsilon)
            Fv[lower_bool] = 6 * v_i_p[lower_bool] * (p_epsilon**2 - 2*p_epsilon + 1) / den
            Fv[~lower_bool] = (8 * v_i_p[~lower_bool]**3 - 12*v_i_p[~lower_bool]**2 + 6*v_i_p[~lower_bool] +2*p_epsilon**3 - 3*p_epsilon**2) / den
            int_Fv[lower_bool] = 3 * v_i_p[lower_bool]**2 * (p_epsilon**2 - 2*p_epsilon + 1) / den
            int_Fv[~lower_bool] = (2 * v_i_p[~lower_bool]**4 - 4*v_i_p[~lower_bool]**3 + 3*v_i_p[~lower_bool]**2 + v_i_p[~lower_bool]*(2*p_epsilon**3 - 3*p_epsilon**2) +0.5*p_epsilon**3 - (3/8)*p_epsilon**4) / den
        else:
            den = p_epsilon - 1
            Fv[lower_bool] = 2 * v_i_p[lower_bool] * np.log(p_epsilon) / den
            Fv[~lower_bool] = (2 * v_i_p[~lower_bool] * (np.log(2 * v_i_p[~lower_bool]) - 1) + p_epsilon) / den
            int_Fv[lower_bool] = np.square(v_i_p[lower_bool]) * np.log(p_epsilon) / den
            int_Fv[~lower_bool] = (4 * np.square(v_i_p[~lower_bool]) * (
                        2 * np.log(2 * v_i_p[~lower_bool]) - 3) + 8 * p_epsilon * v_i_p[
                                       ~lower_bool] - p_epsilon ** 2) / (8 * den)
        ra_bids = np.minimum(1, p_epsilon + v_i_p * Fv - int_Fv)
        ra_participation_utility = v_i_d - ra_bids + v_i_p * Fv
        ra_participate_bool = ra_participation_utility > 0
        ra_num_participating = np.sum(ra_participate_bool)
        ra_rates[i] = 100 * ra_num_participating / num_agents
        ra_avgerage_bids[i] = np.mean(ra_bids[ra_participate_bool])  # eliminate non-participants from computation

        # simple threshold participation rate
        st_participate_bool = v_i_d >= p_epsilon
        st_num_participating = np.sum(st_participate_bool)
        st_rates[i] = 100 * st_num_participating / num_agents
        st_bids = st_participate_bool * p_epsilon
        st_average_bids[i] = np.mean(st_bids[st_bids > 0])

        i += 1

    lab = 'beta' if beta else 'uniform'
    plt.figure(1)
    plt.plot(epsilons, ra_rates, label='SIRA')
    plt.plot(epsilons, st_rates, label='Reserve Thresholding')
    plt.grid()
    plt.xlabel('Epsilon Price $p_\epsilon$ Threshold', fontsize="15", fontweight='bold')
    plt.ylabel('Agent Participation Rate (%)', fontsize="15", fontweight='bold')
    plt.legend(loc='best', fontsize="15")
    plt.ylim([0, 100])
    plt.xlim([0, 1])
    filename = 'regauc_participation_rates_{}_{}.png'.format(lab, lambda_max)
    plt.savefig(filename, dpi=500)
    plt.show()

    plt.figure(2)
    plt.plot(epsilons, ra_avgerage_bids, label='SIRA')
    plt.plot(epsilons, st_average_bids, label='Reserve Thresholding')
    plt.grid()
    plt.xlabel('Epsilon Price $p_\epsilon$ Threshold', fontsize="15", fontweight='bold')
    plt.ylabel('Average Participating Agent Bid', fontsize="15", fontweight='bold')
    plt.legend(loc='best', fontsize="15")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    filename = 'regauc_average_bid_{}_{}.png'.format(lab, lambda_max)
    plt.savefig(filename, dpi=500)
    plt.show()


if __name__ == '__main__':
    lam = 1/2
    regauc_sim(lam, beta=False)
