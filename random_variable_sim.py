import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    beta = False
    num_agents = int(5e7) if not beta else int(1e8)
    p_epsilon = 0.5
    if beta:
        V = np.random.beta(2, 2, size=num_agents)
        V = V[V >= p_epsilon]
        num_agents = len(V)
    else:
        V = np.random.uniform(p_epsilon, 1, size=num_agents)
    Lam = np.random.uniform(0, 0.5, size=num_agents)

    v_i_p = V*Lam
    count, bins_count = np.histogram(v_i_p, bins='auto')
    pdf = count / (sum(count) * np.diff(bins_count))
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    x = np.linspace(0, 1 / 2, int(len(count)), endpoint=True)
    x_1 = x[x <= p_epsilon / 2]
    x_2 = x[x >= p_epsilon / 2]

    if beta:
        beta_cdf = lambda x: 3*x**2 - 2*x**3
        y_pdf1 = np.ones(len(x_1)) * 6*(p_epsilon**2 - 2*p_epsilon + 1) / (1 - beta_cdf(p_epsilon))
        y_pdf2 = 6*(4*x_2**2 - 4*x_2 + 1) / (1 - beta_cdf(p_epsilon))
        y_cdf1 = 6 * x_1 * (p_epsilon**2 - 2*p_epsilon + 1) / (1 - beta_cdf(p_epsilon))
        y_cdf2 = (2*x_2*(4*x_2**2 - 6*x_2 + 3) + p_epsilon**2 * (2*p_epsilon - 3)) / (1 - beta_cdf(p_epsilon))
    else:
        y_pdf1 = np.ones(len(x_1)) * 2 * np.log(p_epsilon) / (p_epsilon - 1)
        y_pdf2 = 2 * np.log(2 * x_2) / (p_epsilon - 1)
        y_cdf1 = 2 * x_1 * np.log(p_epsilon) / (p_epsilon - 1)
        y_cdf2 = (2 * x_2 * (np.log(2 * x_2) - 1) + p_epsilon) / (p_epsilon - 1)

    # plotting PDF and CDF
    lab = 'beta' if beta else 'uniform'
    max_pdf = np.max(y_pdf1)
    pdf1_label = '$f_{v_i^p}(v) = \\frac{6(p_\epsilon^2 - 2p_\epsilon + 1)}{1 - F_\\beta(p_\epsilon)}$' if beta else '$f_{v_i^p}(v) = \\frac{2\ln(p_\epsilon)}{p_\epsilon - 1}$'
    pdf2_label = '$f_{v_i^p}(v) = \\frac{6(4v^2 - 4v + 1)}{1 - F_\\beta(p_\epsilon)}$' if beta else '$f_{v_i^p}(v) = \\frac{2\ln(2v)}{p_\epsilon - 1}$'
    cdf1_label = '$F_{v_i^p}(v) = \\frac{6v(p_\epsilon^2 - 2p_\epsilon + 1)}{1 - F_\\beta(p_\epsilon)}$' if beta else '$F_{v_i^p}(v) = \\frac{2v\ln(p_\epsilon)}{p_\epsilon - 1}$'
    cdf2_label = '$F_{v_i^p}(v) = \\frac{2v(4v^2 - 6v + 3) + p_\epsilon^2(2p_\epsilon - 3)}{1 - F_\\beta(p_\epsilon)}$' if beta else '$F_{v_i^p}(v) = \\frac{2v(\ln(2v)-1) + p_\epsilon}{p_\epsilon - 1}$'

    plt.figure(1)
    plt.plot(bins_count[1:], pdf, color="red", label="Simulated PDF")
    plt.plot(x_1, y_pdf1, color="blue", label=pdf1_label)
    plt.plot(x_2, y_pdf2, '--', color="blue", label=pdf2_label)
    plt.xlabel('Random Value Product $v$', fontsize="14", fontweight='bold')
    plt.ylabel('Probability Density', fontsize="14", fontweight='bold')
    plt.xlim([0, 0.5])
    plt.ylim([0, max_pdf*1.05])
    plt.legend(fontsize="14")
    plt.grid()
    filename = 'regauc_pdf_{}_pe_{}.png'.format(lab, p_epsilon)
    plt.savefig(filename, dpi=500)
    plt.show()

    plt.figure(2)
    plt.plot(bins_count[1:], cdf, label="Simulated CDF", color='red')
    plt.plot(x_1, y_cdf1, '--', color="blue", label=cdf1_label)
    plt.plot(x_2, y_cdf2, ':', color="blue", label=cdf2_label)
    plt.xlabel('Random Value Product $v$', fontsize="15", fontweight='bold')
    plt.ylabel('Probability', fontsize="15", fontweight='bold')
    plt.xlim([0, 0.5])
    plt.ylim([0, 1.025])
    plt.legend(fontsize="14")
    plt.grid()
    filename = 'regauc_cdf_{}_pe_{}.png'.format(lab, p_epsilon)
    plt.savefig(filename, dpi=500)
    plt.show()
