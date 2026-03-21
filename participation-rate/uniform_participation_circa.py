import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm


def regauc_sim(lambda_max):
    epsilons = np.linspace(0.01, 1, 99, endpoint=False)
    ra_rates = np.empty_like(epsilons)
    st_rates = np.empty_like(epsilons)

    for i, eps in enumerate(tqdm(epsilons)):

        # --- CIRCA participation: P(v_d + int_F(v_p) > eps) ---
        # Simplification: u > 0 iff v_d - b* + v_p*F(v_p) > 0
        # Since b* = eps + v_p*F(v_p) - int_F(v_p), this reduces to v_d + int_F(v_p) > eps
        def int_Fv(vp):
            lo = vp <= eps / 2
            safe_2vp = 1 if lo else 2 * vp
            den = eps - 1
            if lo:
                return vp**2 * np.log(eps) / den
            else:
                return (4 * vp**2 * (2 * np.log(safe_2vp) - 3)
                        + 8 * eps * vp - eps**2) / (8 * den)

        def integrand(lam, V):
            vp = V * lam
            vd = V * (1 - lam)
            return 1.0 if (vd + int_Fv(vp)) > eps else 0.0

        # Double integral over V in [0,1] and lambda in [0, lambda_max]
        # Normalise by the area of the domain (1 * lambda_max)
        result, _ = integrate.dblquad(
            integrand,
            0, 1,                   # V limits
            0, lambda_max,          # lambda limits
        )
        ra_rates[i] = 100 * result / lambda_max

        # --- Reserve thresholding: P(v_d >= eps) ---
        # v_d = V*(1-lambda), V~U[0,1], lambda~U[0,lambda_max]
        # P(V*(1-lambda) >= eps) = double integral of 1_{V(1-lam)>=eps} dV dlam / lambda_max
        def st_integrand(lam, V):
            return 1.0 if V * (1 - lam) >= eps else 0.0

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
    fig.savefig(f'regauc_participation_rates_uniform_{lambda_max}.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    regauc_sim(lambda_max=1/2)
