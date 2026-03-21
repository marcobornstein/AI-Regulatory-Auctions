import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# --- Beta(2,2) standard PDF and CDF ---
def f_beta_std(x):
    return 6 * x * (1 - x)

def F_beta_std(x):
    return 3*x**2 - 2*x**3

# --- Product distribution PDF and CDF (from corollary) ---
def f_vp(z, pe):
    """Piecewise PDF of v^p = V_i * lambda_i"""
    norm = 1 - F_beta_std(pe)
    bp = pe / 2
    if z <= 0 or z > 0.5:
        return 0.0
    if z <= bp:
        return 6 * (pe**2 - 2*pe + 1) / norm
    else:
        return 6 * (4*z**2 - 4*z + 1) / norm

def F_vp(z, pe):
    """Piecewise CDF of v^p = V_i * lambda_i"""
    norm = 1 - F_beta_std(pe)
    bp = pe / 2
    if z <= 0:
        return 0.0
    if z <= bp:
        return 6 * z * (pe**2 - 2*pe + 1) / norm
    elif z <= 0.5:
        return (2*z*(4*z**2 - 6*z + 3) + pe**2*(2*pe - 3)) / norm
    return 1.0

# --- Integrand: z * f(z) * (1 - F(z)) ---
def integrand(z, pe):
    if z <= 0:
        return 0.0
    return z * f_vp(z, pe) * (1 - F_vp(z, pe))

# --- Expected bid ---
def expected_bid(pe):
    """E[b*] = p_epsilon + integral from 0 to 1/2 of z*f(z)*(1-F(z))dz"""
    bp = pe / 2
    vec_integrand = np.vectorize(lambda z: integrand(z, pe))
    # Split at breakpoint pe/2 for numerical accuracy
    I1, _ = integrate.quad(vec_integrand, 1e-10, bp)
    I2, _ = integrate.quad(vec_integrand, bp, 0.5)
    return pe + I1 + I2

# --- Compute over range of p_epsilon ---
pe_values = np.linspace(0.05, 0.95, 400)
eb_values = np.array([expected_bid(pe) for pe in pe_values])

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(pe_values, eb_values, color='steelblue', linewidth=2, label=r'Circa Bid (Beta)')
ax.plot(pe_values, pe_values, color='gray', linewidth=1, linestyle='--', label=r'Reserve Thresholding Bid ($p_\epsilon$)')
ax.fill_between(pe_values, pe_values, eb_values, alpha=0.15, color='steelblue', label='Bid Surplus')

ax.set_xlabel(r'Epsilon Price Threshold $p_\epsilon$', fontsize="14", fontweight='bold')
ax.set_ylabel(r'Agent Expected Bid $\mathbb{E}[\hat{b}_i^*]$', fontsize="14", fontweight='bold')
ax.legend(fontsize=14)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('expected_bid_beta.png', dpi=150)
plt.show()
