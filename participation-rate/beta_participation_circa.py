import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm


def regauc_sim(lambda_max):
    epsilons = np.linspace(0.01, 1, 99, endpoint=False)
    ra_rates = np.empty_like(epsilons)
    st_rates = np.empty_like(epsilons)

    beta_pdf = lambda V: 6 * V * (1 - V)           # Beta(2,2) PDF on [0,1]
    beta_cdf = lambda x: 3*x**2 - 2*x**3

    for i, eps in enumerate(tqdm(epsilons)):

        def int_Fv(vp):
            den = 1 - beta_cdf(eps)
            if vp <= eps / 2:
                return 3 * vp**2 * (eps**2 - 2*eps + 1) / den
            else:
                return (2*vp**4 - 4*vp**3 + 3*vp**2
                        + vp*(2*eps**3 - 3*eps**2)
                        + 0.5*eps**3 - (3/8)*eps**4) / den

        # --- CIRCA: integral of 1_{v_d + int_F(v_p) > eps} weighted by Beta PDF ---
        def ra_integrand(lam, V):
            vp = V * lam
            vd = V * (1 - lam)
            return beta_pdf(V) * (1.0 if (vd + int_Fv(vp)) > eps else 0.0)

        result, _ = integrate.dblquad(
            ra_integrand,
            0, 1,           # V limits
            0, lambda_max,  # lambda limits
        )
        ra_rates[i] = 100 * result / lambda_max   # normalise by lambda domain only;
                                                   # beta_pdf integrates to 1 over [0,1]

        # --- Reserve thresholding: P(v_d >= eps) weighted by Beta PDF ---
        def st_integrand(lam, V):
            return beta_pdf(V) * (1.0 if V * (1 - lam) >= eps else 0.0)

        st_result, _ = integrate.dblquad(
            st_integrand,
            0, 1,
            0, lambda_max,
        )
        st_rates[i] = 100 * st_result / lambda_max

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(epsilons, ra_rates, color='steelblue', linewidth=2,
            label=r'CIRCA: $u_i(b^*) > 0$')
    ax.plot(epsilons, st_rates, color='gray', linewidth=2,
            linestyle='--', label=r'Reserve Thresholding: $v_i^d \geq p_\epsilon$')
    ax.fill_between(epsilons, st_rates, ra_rates,
                    alpha=0.20, color='steelblue', label='Participation Surplus')
    ax.set_xlabel(r'Epsilon Price $p_\epsilon$ Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Agent Participation Rate (%)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=14)
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'regauc_participation_rates_beta_{lambda_max}.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    regauc_sim(lambda_max=1/2)
