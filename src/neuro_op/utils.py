# Useful functions for neuro_op dynamics implementation
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as st


# All-lowercase glossary of terms & abbreviations used in the neuro_op package
glossary = {
    "dist": "Distribution",
    "llf": "Likelihood function",
    "mle": "Maximum likelihood estimate",
    "mu": "Mean",
    "pdf": "Probability density function",
    "ppd": "Posterior predictive distribution",
    "probs": "Probabilities",
    "sd": "Standard deviation",
}


def build_random_network(N_nodes, N_neighbours):
    """
    Return directed graph of N_nodes with random connections.
    At the start, each node has, on average, N_neighbours connections, each connection is bidirectional and of weight 1.
    """

    p = N_neighbours / (
        N_nodes - 1
    )  # probability of connection; '-1' to exclude self-connections

    G = nx.gnp_random_graph(N_nodes, p).to_directed()

    return G


def build_stochastic_block_model(N_nodes, N_blocks, N_neighbours, p_in=0.5, p_out=0.1):
    """
    Return a directed stochastic block model graph.

    Build a graph of N_blocks equally-sized blocks with p_in probability of connection within a block and p_out probability of connection between blocks.
    """

    block_sizes = [N_nodes // N_blocks] * N_blocks
    block_sizes[-1] += N_nodes % N_blocks  # add remainder to last block
    p = []
    for i in range(N_blocks):
        p += [[p_out] * N_blocks]
        p[i][i] = p_in
    p = np.array(p) / np.sum(p)

    # Adjust p to match desired mean degree
    G_tmp = nx.stochastic_block_model(block_sizes, p, directed=True)
    mean_degree = np.sum([G_tmp.degree(n) for n in G_tmp.nodes()]) / (2 * len(G_tmp))
    p = p * N_neighbours / mean_degree

    G = nx.stochastic_block_model(block_sizes, p, directed=True)

    # Small plausibility checks
    assert len(G) == N_nodes
    print("Mean degree: ", np.sum([G.degree(n) for n in G.nodes()]) / (2 * len(G)))
    return G


def logpdf_to_pdf(logprobs):
    """
    Returns array of relative log probabilities as normalized relative probabilities.

    Keyword arguments:
    logprobs : iterable
        array of log probabilities
    """

    probs = np.exp(
        logprobs - np.max(logprobs)
    )  # shift logprobs before np.exp so that exp(logprobs)>0; needed due to non-normalized log-space Bayesian update

    return probs / np.sum(probs)


def dist_binning(llf, params, N_bins=50, range=(-20, 20)):
    """
    Returns a logpdf method's probability mass binned into N equally-spaced bins.

    Returns normalized numpy array of probabilities to sample value in corresponding bin.
    'Integrates' the logpdf function by summing over exponentiated logpdf values at bin margins.

    Keyword arguments:
    logpdf : iterable
        discrete log-probabilities to be binned
    N_bins : int
        number of bins to be returned
    range : tuple
        interval over which binning is performed
    """

    Q_bins = np.linspace(range[0], range[1], N_bins + 1)
    # Q_logpdf = llf.logpdf(**params, x=Q_bins)
    # Q = np.exp(Q_logpdf[:-1]) + np.exp(Q_logpdf[1:])
    Q = llf.cdf(Q_bins[1:], **params) - llf.cdf(Q_bins[:-1], **params)

    return Q / np.sum(Q)


def get_MLE_Grid(log_probs, mu_arr):
    """
    Returns MLE of mu, or, for multiple maxima, the "most central" one.
    """

    return mu_arr[[i[len(i) // 2] for i in np.where(log_probs == np.max(log_probs))][0]]


def kl_divergence(P, Q):
    """
    Returns Kullback-Leibler divergence between two identically binned discrete probability distributions.

    Returns Kullback-Leibler divergence in base b=e (i.e., nats).
    Usually, P is the to-be-tested (and/or data-based) distribution and Q the reference distribution.

    If you want to use samples as basis of your inputs, normalize and bin them via
    > inputs = np.histogram(samples, bins=N_bins, range=(sample_space_min, sample_space_max)
    > inputs = inputs[0] / np.sum(inputs[0])

    Keyword arguments:
    P : iterable
        reference discrete probability distribution
    Q : iterable
        recorded discrete probability distribution
    """

    # Normalize P and Q
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # kind of wackily add machine epsilon to all elements of Q to avoid 'divided by zero' errors
    epsilon = 7.0 / 3 - 4.0 / 3 - 1
    Q += epsilon
    P += epsilon
    terms = P * np.log(P / Q)

    # assure that KL-div is 0 for (P==0 && Q>0)
    terms[(P / Q == 0)] = 0

    return np.sum(terms)


def get_p_distances(param_node, param_ref, p=1, p_inv=1):
    """
    Returns an average distance between an array of nodes' inferred parameters and a reference parameter.

    Returns a distance between an array of node-inferred generative model parameters and a reference value.
    Usually, the reference value of interest is the parameter encoding/generating the real world data
    (e.g., the mean of a Gaussian distribution).
    Usually, the array of MLEs is created from nodes' posterior predictive distributions (PPDs).

    If p_inv = 1/p, this function returns the p-norm distances.
    If p_inv = 1, the return value is usually multiplied by 1/len(mu_nodes) to obtain the mean distance.
    For sum of linear distances, choose p = 1, p_inv = 1  .
    For sum of quadratic distances, choose p = 2, p_inv = 1  .

    Keyword arguments:
    praram_node : iterable
        array of nodes' inferred parameters
    param_ref : iterable
        array of real parameter value(s)
    p : float
        value by which each difference is raised, usually 1 (linear distance) or 2 (squared distance)
    p_inv : float
        value by which the sum of differences is raised, usually 1 (mean) or 1/p (p-norm)
    """

    return np.sum(np.abs(param_node - param_ref) ** p) ** p_inv


def postrun_Mu_ConjMu(
    mu_prior,
    sd_prior,
    sd_llf,
    diary_in,
):
    """Calculate posterior param.s of a NodeConjMu with some diary_in (subset) as input."""
    x_in = np.array(diary_in)[:, 0]
    # t_in = np.array(diary_in)[:, 2]
    mu_post = np.zeros_like(x_in)
    sd_post = np.zeros_like(x_in)
    mu_post[-1], sd_post[-1] = mu_prior, sd_prior
    for i, _ in enumerate(mu_post):
        mu_post[i] = (x_in[i] * sd_post[i - 1] ** 2 + mu_post[i - 1] * sd_llf**2) / (
            sd_post[i - 1] ** 2 + sd_llf**2
        )
        sd_post[i] = (1 / sd_post[i - 1] ** 2 + 1 / sd_llf**2) ** (-0.5)

    return mu_post, sd_post


def postrun_kld_ConjMu(mu_ppd, sd_ppd, mu_real=0, sd_real=1, N_bins=201, range=(-5, 5)):
    """Calculate Kullback-Leibler divergences between normal distributions of estimated and real parameter values."""
    P = dist_binning(st.norm, {"loc": mu_real, "scale": sd_real}, N_bins, range)
    kl_divs = [
        kl_divergence(
            Q=dist_binning(st.norm, {"loc": mu, "scale": sd}, N_bins, range), P=P
        )
        for mu, sd in zip(mu_ppd, sd_ppd)
    ]
    return kl_divs


def normal_entropy(x, loc, scale):
    """Approximated entropy of a normal distribution at a specific value x."""
    return 0.5 * np.log(2 * np.pi * scale**2 + 1e-6) + 0.5 * ((x - loc) / scale) ** 2


def plot_setup(fig_dpi=100, save_dpi=1200):
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.dpi"] = fig_dpi
    plt.rcParams["figure.figsize"] = (3, 1.5)
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.dpi"] = save_dpi
    plt.rcParams["savefig.transparent"] = True
    return None


def get_colorcycle(N_colors, cmap="viridis"):
    colormap = plt.get_cmap(cmap)
    return [colormap(i) for i in np.linspace(1 / N_colors, 1 - 1 / N_colors, N_colors)]
