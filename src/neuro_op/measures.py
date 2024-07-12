# Implementation of network dynamics measures, e.g.,
# distribution distances (Kullback-Leibler divergence (KLD), mean (quad.) distances, ...)
import numpy as np
import scipy.stats as st

from .utils import *


def ppd_Gaussian_mu(llf_nodes, beliefs, logprobs, sigma, N_samples=1000):
    """
    Simulate predictions using the whole posterior, with the underlying likelihood logprobability funtion (llh) being Gaussian.

    Posterior predictive distribution (PPD) sampling first samples paramter values of the estimand from the posterior.
    Then these sampled parameter values will be used in llh  to sample predictions.
    Thereby, theppd includes all the uncertainty (i.e., model parameter value uncertainty (from posterior) & generative uncertainty (model with given parameter values creating data stochastically).

    Keyword arguments:
    llf_nodes : scipy.stats function
        Likelihood function (llf) of nodes
    beliefs : iterable
        possible parameter values into which a node may hold belief
    logprobs : iterable
        log probabilities, corresponding to 'beliefs' array
    sigma : float
        Node-supplied standard deviation of the Gaussian likelihood function
    N_samples : int
        number of to-be-drawn likelihood (llh) parameter values and then-sampled predictions; can in principle be split up into two separate parameters (one for parameter sampling, one for prediction sampling)
    """

    # Transform potentially non-normalized log probabilities to normalized probabilities.
    probs = logpdf_to_pdf(logprobs)

    # Sample parameter values proportional to 'probs'.
    params = dict(
        loc=np.random.choice(beliefs, p=probs, size=N_samples),
        scale=sigma,
    )

    # Generate predictions using the llh method.
    return llf_nodes.rvs(**params)


def ppd_distances_Gaussian(
    llf_nodes,
    llf_world,
    beliefs,
    nodes,
    world,
    sample_bins=50,
    sample_range=(-20, 20),
    p_distance_params=[],
):
    """
     Return approximated distances between system nodes'ppds and world state's distribution and binning used during approximation.

     Approximates distance between system nodes' posterior predictive distributions (PPDs) and world state's distribution.
    ppds and world state distributions are approximated by histogram-binning posterior predictive samples of each distribution.
     Then, the wanted distance (KL divergence or p-distance) is calculated between each node distribution and the world distribution.

     Keyword arguments:
     llf_nodes : scipy.stats function
         Likelihood function (llf) of nodes
     llf_world : scipy.stats function
         Likelihood function (llf) of world
     beliefs : iterable
         Possible parameter values into which a node may hold belief
     nodes : list of Node objects
         Objects for which to calculate distances
     world : Node object
         Node stroing & providing the actual/"real" state of the world
     sample_bins : int
         Number of bins used in histogram binning of posterior predictive samples
     samlpe_range : tuple
         Interval over which binning is performed
    """

    ppd_bins = np.linspace(sample_range[0], sample_range[1], sample_bins + 1)

    # Generate posterior predictive distributions (PPDs) for each node by generating ppd samples and binning them into histograms
    ppd_samples = [
        ppd_Gaussian_mu(
            llf_nodes,
            beliefs,
            node.log_probs,
            node.params_node["scale"],
            N_samples=1000,
        )
        for node in nodes
    ]

    ppds = [
        np.histogram(i, bins=sample_bins, range=sample_range)[0] for i in ppd_samples
    ]  # createppd approximations via sampling and binning into histograms

    if world.diary_out:
        ppd_world_out = np.histogram(  # worldppd from all information shared to the network. Also stores binning used for allppds.
            np.array(world.diary_out)[:, 0], bins=sample_bins, range=sample_range
        )
        ppd_world_out = ppd_world_out[0] / np.sum(
            ppd_world_out[0]
        )  # normalize world_outppd
    else:
        ppd_world_out = np.zeros(sample_bins)

    ppd_world_true = dist_binning(
        llf_world,
        world.params_node,
        sample_bins,
        sample_range,
    )

    # Get MLEs of each node'sppd -- note this implementation is not robust toppds with multiple peaks of same height
    argmax = [np.where(i == np.max(i))[0] for i in ppds]
    argmax = [i[len(i) // 2] for i in argmax]
    mu_nodes = [(ppd_bins[i] + ppd_bins[i + 1]) / 2 for i in argmax]

    # Get KL-divergences of each node'sppd
    kl_divs = []
    for i in ppds:
        node_ppd = i / np.sum(i)
        kl_divs.append(
            [
                kl_divergence(node_ppd, ppd_world_out),
                kl_divergence(node_ppd, ppd_world_true),
            ]
        )

    # If array for 'get_p_distances' function is not empty , calculate p-distances between each node's MLE and the world's MLE
    p_distances = []
    if p_distance_params:
        # First approach: Go for MLE comparisons
        argmax = np.argmax(ppd_world_out)
        mu_world_out = (ppd_bins[argmax] + ppd_bins[argmax + 1]) / 2
        argmax = np.argmax(ppd_world_true)
        mu_world_true = (ppd_bins[argmax] + ppd_bins[argmax + 1]) / 2

        for p in p_distance_params:
            p_distances.append(
                [
                    [
                        get_p_distances(mu_i, mu_world_out, p=p[0], p_inv=p[1])
                        for mu_i in mu_nodes
                    ],
                    [
                        get_p_distances(mu_i, mu_world_true, p=p[0], p_inv=p[1])
                        for mu_i in mu_nodes
                    ],
                ]
            )

    return (mu_nodes, kl_divs, p_distances)


def ppd_distances_Laplace(
    llf_nodes,
    llf_world,
    nodes,
    world,
    sample_bins=50,
    sample_range=(-20, 20),
    p_distance_params=[],
):
    """Sample MLEs, distance measures for NodeConjMu system."""

    # Create posterior predictive distribution for each node and world
    ppd_nodes = [
        np.histogram(
            st.norm.rvs(
                loc=llf_nodes.rvs(**node.params_node, size=1000),
                scale=node.sd_llf,
            ),
            bins=sample_bins,
            range=sample_range,
        )[0]
        for node in nodes
    ]

    ppd_world = dist_binning(
        llf_world,
        world.params_node,
        sample_bins,
        sample_range,
    )

    # Get MLEs of each node
    mu_nodes = [node.params_node["loc"] for node in nodes]

    # Get KL-divergences of each node'sppd
    kl_divs = [kl_divergence(i, ppd_world) for i in ppd_nodes]

    # If p_distance_params given, calculate p-distances
    p_distances = []
    if p_distance_params:
        for p in p_distance_params:
            p_distances.append(
                [
                    get_p_distances(mu_i, world.params_node["loc"], p=p[0], p_inv=p[1])
                    for mu_i in mu_nodes
                ]
            )
    return (mu_nodes, kl_divs, p_distances)
