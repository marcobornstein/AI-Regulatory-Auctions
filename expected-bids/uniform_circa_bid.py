import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(z, pe):
    """PDF of v^p for uniform distribution"""
    if z <= pe / 2:
        return 2 * np.log(pe) / (pe - 1)
    else:
        return 2 * np.log(2 * z) / (pe - 1)

def F(z, pe):
    """CDF of v^p for uniform distribution"""
    if z <= pe / 2:
        return 2 * z * np.log(pe) / (pe - 1)
    else:
        return (2 * z * (np.log(2 * z) - 1) + pe) / (pe - 1)

def integrand(z, pe):
    """z * f(z) * (1 - F(z))"""
    if z <= 0:
        return 0.0
    return z * f(z, pe) * (1 - F(z, pe))

def expected_bid(pe):
    """E[b*] = p_epsilon + integral from 0 to 1/2 of z*f(z)*(1-F(z))dz"""
    vec_integrand = np.vectorize(lambda z: integrand(z, pe))
    # Split at breakpoint pe/2 for numerical accuracy
    I1, _ = integrate.quad(vec_integrand, 1e-10, pe / 2)
    I2, _ = integrate.quad(vec_integrand, pe / 2, 0.5)
    return pe + I1 + I2

# p_epsilon must be in (0,1) but practically agents need p_epsilon < 1
# Also p_epsilon > 0 for log to be defined; use range (0.05, 0.95)
pe_values = np.linspace(0.05, 0.95, 400)
eb_values = np.array([expected_bid(pe) for pe in pe_values])

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(pe_values, eb_values, color='steelblue', linewidth=2, label=r'CIRCA Bid (Uniform)')
ax.plot(pe_values, pe_values, color='gray', linewidth=1, linestyle='--', label=r'Reserve Thresholding Bid ($p_\epsilon$)')
ax.fill_between(pe_values, pe_values, eb_values, alpha=0.15, color='steelblue', label='Bid Surplus')

ax.set_xlabel(r'Epsilon Price Threshold $p_\epsilon$', fontsize="14", fontweight='bold')
ax.set_ylabel(r'Agent Expected Bid $\mathbb{E}[\hat{b}_i^*]$', fontsize="14", fontweight='bold')
ax.legend(fontsize=14)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('expected_bid_uniform.png', dpi=150)
plt.show()
