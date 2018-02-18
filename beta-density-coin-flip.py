from scipy.stats import beta

import matplotlib.pyplot as plt
import numpy as np

outcomes = [1, 0, 0, 1, 0, 0, 0]


def get_beta_posterior_parameters(head, tail, v_h=None, v_t=None):
    """
    Gets the parameters for beta distribution
    """
    v_h = v_h if v_h else 1
    v_t = v_t if v_t else 1
    return v_h + head, v_t + tail


def plot_beta_distribution():
    head = 0
    tail = 0
    x = np.linspace(beta.ppf(0.00, 1, 1),
                    beta.ppf(1.00, 1, 1), 200)
    plt.subplot(2, 4, 1)
    plt.plot(x, beta.pdf(x, 1, 1), 'r-', lw=5, label='beta pdf')
    counter = 2

    for outcome in outcomes:
        head = head + outcome
        tail = tail + (1 - outcome)
        alpha, b = get_beta_posterior_parameters(head, tail)
        x = np.linspace(beta.ppf(0.00, alpha, b),
                        beta.ppf(1.00, alpha, b), 200)
        plt.subplot(2, 4, counter)
        plt.plot(x, beta.pdf(x, alpha, b), 'r-', lw=5, label='beta pdf')
        counter += 1

    # plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    plot_beta_distribution()
