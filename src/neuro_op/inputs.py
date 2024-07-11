# Reference input dictionaries for the neuro_op package run functions
import numpy as np
import scipy.stats as st

from .utils import build_random_network


input_ref_Grid = dict(
    # Reference input for 'run_Grid' function. For description of contents, see 'run_model' function docstring.
    G=build_random_network(N_nodes=100, N_neighbours=5),  # networkx graph object
    mu_arr=np.linspace(-10, 10, 201),  # beliefs considered by each node
    sd_arr=np.linspace(0, 10, 101)[
        1:
    ],  # standard deviations of beliefs considered by each node
    log_priors=np.zeros((201, 100)),  # Prior log-probabilities of nodes
    llf_nodes=st.norm,  # Likelihood function (llf) of nodes, Gaussian by default
    llf_world=st.norm,  # Likelihood function (llf) of to-be-approximated world state, Gaussian by default
    params_world=dict(  # Likelihood function (llf) parameters of to-be-approximated world state, Gaussian by default
        loc=0,
        scale=1,
    ),
    h=1,  # Rate of external information draw events
    r=1,  # Rate of edge information exchange events
    t0=0,  # Start time of simulation
    t_max=50,  # End time of simulation
    t_sample=1,  # Periodicity for which mu_nodes and KLD are sampled
    sample_range=(
        -10,
        10,
    ),  # Interval over which distance measure distributions are considered
    sample_bins=101,  # Number of bins used in distance measures
    sampling=True,
)

input_ref_GridMu = dict(
    G=build_random_network(N_nodes=100, N_neighbours=5),
    mu_arr=np.linspace(-10, 10, 201),
)

input_ref_Param = dict(
    G=build_random_network(N_nodes=100, N_neighbours=5),  # networkx graph object
    llf_nodes=st.norm,  # Likelihood function (llf) of nodes, Gaussian by default
    llf_world=st.norm,  # Likelihood function (llf) of world, Gaussian by default
    params_node=dict(  # Parameter priors of nodes (mu and associated uncertainty (standard deviation)), Gaussian by default
        loc=0,
        scale=10,
    ),
    sd_llf=1,  # Standard deviation of the likelihood function (llf) of nodes, assumed known & static
    params_world=dict(  # Likelihood function (llf) parameters of world, Gaussian by default
        loc=0,
        scale=1,
    ),
    h=1,  # Rate of external information draw events
    r=1,  # Rate of edge information exchange events
    t0=0,  # Start time of simulation
    t_max=50,  # End time of simulation
    t_sample=0.5,  # Periodicity for which samples and distance measures (KL-div, p-distance) are taken
    sample_bins=201,  # Number of bins used in distance measures
    sample_range=(
        -5,
        5,
    ),  # Interval over which distance measure distributions are considered
    sampling=True,  # Switch for sampling
)
