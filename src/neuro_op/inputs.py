# Reference input dictionaries for the neuro_op package run functions
import numpy as np
import scipy.stats as st

from .utils import build_random_network


input_ref_Grid = dict(
    G=build_random_network(N_nodes=100, N_neighbours=5),  # networkx graph object
    llf_nodes=st.norm,  # Likelihood function (llf) of nodes
    mu_arr=np.linspace(-10, 10, 201),  # means which nodes consider
    sd_arr=np.linspace(0, 10, 101)[1:],  # standard deviations which nodes consider
    log_priors=np.zeros((201, 100)),  # Nodes' prior log-probabilities
    llf_world=st.norm,  # Likelihood function (llf) of to-be-approximated world state, Gaussian by default
    params_world=dict(  # Likelihood function (llf) parameters of to-be-approximated world state, Gaussian by default
        loc=0,
        scale=1,
    ),
    h=1,  # "World shares information" rate per node
    r=1,  # "Neighbours share information" rate per node
    t0=0,  # Start time of simulation
    t_max=50,  # End time of simulation
    t_sample=1,  # Sampling periodicity
    sample_range=(-10, 10),
    sample_bins=201,
    sampling=True,
    init_rngs=False,  # Re-initialize random number generators for a reproducible run
    seed=False,
)


input_ref_GridMu = dict(
    G=build_random_network(N_nodes=100, N_neighbours=5),
    llf_nodes=st.norm,
    mu_arr=np.linspace(-10, 10, 201),
    log_priors=np.zeros(201),
    sd_llf=1,
    llf_world=st.norm,
    mu_world=0,
    sd_world=1,
    h=1,
    r=1,
    t0=0,
    t_max=50,
    t_sample=1,
    sample_range=(-10, 10),
    sample_bins=201,
    sampling=True,
    init_rngs=False,
    seed=False,
)


input_ref_ConjMu = dict(
    G=build_random_network(N_nodes=150, N_neighbours=5),  # networkx graph object
    llf_nodes=st.norm,  # Likelihood function (llf) of nodes, Gaussian by default
    params_node=dict(  # Parameter priors of nodes (mu and associated uncertainty (standard deviation)), Gaussian by default
        loc=0,
        scale=10,
    ),
    sd_llf=1,  # Standard deviation of the likelihood function (llf) of nodes, assumed known & static
    llf_world=st.norm,  # Likelihood function (llf) of world, Gaussian by default
    params_world=dict(  # Likelihood function (llf) parameters of world, Gaussian by default
        loc=0,
        scale=1,
    ),
    h=1,  # Rate of external information draw events
    r=1,  # Rate of edge information exchange events
    t0=0,  # Start time of simulation
    t_max=100,  # End time of simulation
    t_sample=1,  # Periodicity for which samples and distance measures (KL-div, p-distance) are taken
    sampling=True,  # Whether to sample and calculate distances
    init_rngs=False,  # Re-initialize random number generators for a reproducible run
    seed=False,  # User-defined rng seed
)
