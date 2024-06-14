import time

import scipy.stats as st

from .classes import *
from .measures import *
from .randomness import *
from .utils import *


def run_model_Grid(
    G,
    llf_nodes,
    llf_world,
    params_node,
    params_world,
    beliefs,
    log_priors,
    h,
    r,
    t0,
    t_max,
    #    t_sample,
    #    sample_bins,
    #    sample_range,
    #    p_distance_params,
    progress,
):
    """
    Execute program.
    Get all parameters and initialize nodes (w. belief and log-prior distributions), network graph, and world distribution.
    Then, run simulation until t>=t_max and return simulation results.

    Keyword arguments:
    G : networkx graph object
        Graph of nodes and edges
    llf_nodes : scipy.stats function
        Likelihood function (llf) of nodes
    llf_world : scipy.stats function
        Likelihood function (llf) of world
    params_node : dict
        Parameters defining the likelihood function (llf) of nodes, concerning a Gaussian by default
    params_world : dict
        Parameters defining the likelihood function (llf) of the world, concerning a Gaussian by default
    beliefs : list (floats)
        Possible parameter values into which a Node may hold beliefs in
    log_priors : list (floats)
        Prior log-probabilities of nodes
    h : float
        Rate of external information draw events
    r : float
        Rate of edge information exchange events
    t0 : float
        Start time of simulation
    t_max : float
        End time of simulation
    t_sample : float
        Periodicity for which samples and distance measures (KL-div, p-distance) are taken
    sample_bins : int
        Number of bins used in distance measures
    sample_range : tuple
        Interval over which distance measure distributions are considered
    p_distance_params : list
        List of tuples, each containing two floats, defining the p-distance parameters
    progress : bool
        Whether or not to print sampling times


    Returns:
    Dictioniary containing the following keys and according data after end of simulation:
    nodes : list (Nodes)
        All nodes of the network.
    G : networkx graph object

    beliefs : np.array (floats)
        Array of possible parameter values into which a Node may hold beliefs in.
    world : Node
        Object representing the world.
    N_events : int
        Number of events executed during simulation.
    t_end : float
        End time of simulation.
    mu_nodes : list
        MAP estimates of each node's mu, sampled during run.
        Shape: (#samples, #nodes)
    kl_divs : list
        KL-divergences between each node's PPD and the world's PPD, sampled during run.
        Shape: (#samples, #nodes, 2)
        '2' refers to world_out ([0]) and world_true ([1]) as reference distribution, respectively.
    p_distances : list
        p-distances between each node's MLE and the world's MLE, sampled during run.
        Shape: (#samples, #(p_distance parameter tuples), 2)
        '2' refers to  world_out ([0]) and world_true ([1]) as reference distribution, respectively.
    t_start : str
        Start time and date of simulation.
    t_exec : float
        Simulation duration in seconds.
    seed : int
        Random seed used for simulation.
    """
    starttime = time.time()

    assert len(beliefs) == len(log_priors)

    # Renormalize rates to keep rate per node constant
    h = h * len(G)
    r = r * len(G)

    nodes = [
        NodeNormal(
            node_id=i,
            log_priors=log_priors,
            params_node=params_node,
        )
        for i in range(len(G))
    ]
    world = NodeNormal(
        node_id=-1,
        log_priors=llf_world.logpdf(**params_world, x=beliefs),
        params_node=params_world,
    )
    #    ppd_func = ppd_distances_Gaussian
    #    ppd_in = dict(
    #        llf_nodes=llf_nodes,
    #        llf_world=llf_world,
    #        beliefs=beliefs,
    #        nodes=nodes,
    #        world=world,
    #        sample_bins=sample_bins,
    #        sample_range=sample_range,
    #        p_distance_params=p_distance_params,
    #    )

    # Run simulation...
    N_events = 0
    t = t0
    #    sample_counter = int(t0 / t_sample)
    #    mu_nodes = []
    #    kl_divs = []
    #    p_distances = []

    while t < t_max:
        #        # Sample system PPDs, distance measures (KL-div, p-distance) with periodicity t_sample
        #        if int(t / t_sample) >= sample_counter:
        #            if progress:
        #                print("Sampling at t=", t, "\t, aka", (t / t_max), "\t of runtime.")
        #            sample_counter += 1
        #            sample_mu_nodes, sample_kl_div, sample_p_distances = ppd_func(**ppd_in)
        #            mu_nodes.append(sample_mu_nodes)
        #            kl_divs.append(sample_kl_div)
        #            p_distances.append(sample_p_distances)

        # Information exchange event...
        N_events += 1
        if rng.uniform() < h / (h + r):
            # external information draw event
            node = rng0.choice(nodes)
            node.set_updated_belief(
                llf_nodes,
                beliefs=beliefs,
                info_in=world.get_belief_sample(beliefs, t),
                id_in=world.node_id,
                t_sys=t,
            )

        else:
            # edge event
            chatters = rng0.choice(list(G.edges()))
            sample0 = nodes[chatters[0]].get_belief_sample(beliefs=beliefs, t_sys=t)
            sample1 = nodes[chatters[1]].get_belief_sample(beliefs=beliefs, t_sys=t)
            nodes[chatters[0]].set_updated_belief(
                llf_nodes, beliefs, sample1, chatters[1], t
            )
            nodes[chatters[1]].set_updated_belief(
                llf_nodes, beliefs, sample0, chatters[0], t
            )

        t += st.expon.rvs(scale=1 / (h + r))

    # Sample post-execution system PPDs, distance measures (KL-div, p-distance), if skipped in last iteration
    #    if int(t / t_sample) >= sample_counter:
    #        if progress:
    #            print("Sampling at t=", t)
    #        sample_counter += 1
    #        sample_mu_nodes, sample_kl_div, sample_p_distances = ppd_func(**ppd_in)
    #        mu_nodes.append(sample_mu_nodes)
    #        kl_divs.append(sample_kl_div)
    #        p_distances.append(sample_p_distances)

    return {
        "nodes": nodes,
        "G": G,
        "beliefs": beliefs,
        "world": world,
        "N_events": N_events,
        "t_end": t,
        #        "mu_nodes": mu_nodes,
        #        "kl_divs": kl_divs,
        #        "p_distances": p_distances,
        "t_start": time.strftime("%Y-%m-%d--%H-%M", time.localtime(starttime)),
        "t_exec": time.time() - starttime,
        "seed": RANDOM_SEED,
    }


def run_model_Param(
    G,
    llf_nodes,
    llf_world,
    params_node,
    sd_llf,
    params_world,
    h,
    r,
    t0,
    t_max,
    #    t_sample,
    #    sample_bins,
    #    sample_range,
    #    p_distance_params,
    progress,
):
    """
    Execute program.
    Get all parameters and initialize nodes (w. belief and log-prior distributions), network graph, and world distribution.
    Then, run simulation until t>=t_max and return simulation results.

    Keyword arguments:
    G : networkx graph object
        Graph of nodes and edges
    llf_nodes : scipy.stats function
        Likelihood function (llf) of nodes
    llf_world : scipy.stats function
        Likelihood function (llf) of world
    params_node : dict
        Parameters defining the likelihood function (llf) of nodes, concerning a Gaussian by default
    params_world : dict
        Parameters defining the likelihood function (llf) of the world, concerning a Gaussian by default
    h : float
        Rate of external information draw events
    r : float
        Rate of edge information exchange events
    t0 : float
        Start time of simulation
    t_max : float
        End time of simulation
    t_sample : float
        Periodicity for which samples and distance measures (KL-div, p-distance) are taken
    sample_bins : int
        Number of bins used in distance measures
    sample_range : tuple
        Interval over which distance measure distributions are considered
    p_distance_params : list
        List of tuples, each containing two floats, defining the p-distance parameters
    progress : bool
        Whether or not to print sampling times


    Returns:
    Dictioniary containing the following keys and according data after end of simulation:
    nodes : list (Nodes)
        All nodes of the network.
    G : networkx graph object

    beliefs : np.array (floats)
        Array of possible parameter values into which a Node may hold beliefs in.
    world : Node
        Object representing the world.
    N_events : int
        Number of events executed during simulation.
    t_end : float
        End time of simulation.
    mu_nodes : list
        MAP estimates of each node's mu, sampled during run.
        Shape: (#samples, #nodes)
    kl_divs : list
        KL-divergences between each node's PPD and the world's PPD, sampled during run.
        Shape: (#samples, #nodes, 2)
        '2' refers to world_out ([0]) and world_true ([1]) as reference distribution, respectively.
    p_distances : list
        p-distances between each node's MLE and the world's MLE, sampled during run.
        Shape: (#samples, #(p_distance parameter tuples), 2)
        '2' refers to  world_out ([0]) and world_true ([1]) as reference distribution, respectively.
    """

    # Renormalize rates to keep rate per node constant
    h = h * len(G)
    r = r * len(G)

    # Set up simulation environment (nodes, world, sampling function/inputs)
    nodes = [
        NodeConjMu(node_id=i, params_node=params_node, sd_llf=sd_llf)
        for i in range(len(G))
    ]
    world = NodeConjMu(node_id=-1, params_node=params_world)
    #    ppd_func = ppd_distances_Laplace
    #    ppd_in = dict(
    #        llf_nodes=llf_nodes,
    #        llf_world=llf_world,
    #        nodes=nodes,
    #        world=world,
    #        sample_bins=sample_bins,
    #        sample_range=sample_range,
    #        p_distance_params=p_distance_params,
    #    )

    # Run simulation...
    N_events = 0
    t = t0
    #    sample_counter = int(t0 / t_sample)
    #    mu_nodes = []
    #    kl_divs = []
    #    p_distances = []

    while t < t_max:
        #        # Sample MLEs, distance measures with periodicity t_sample
        #        if int(t / t_sample) >= sample_counter:
        #            if progress:
        #                print("Sampling at t=", t, "\t, aka", (t / t_max), "\t of runtime.")
        #            sample_counter += 1
        #            sample_mu_nodes, sample_kl_div, sample_p_distances = ppd_func(**ppd_in)
        #            mu_nodes.append(sample_mu_nodes)
        #            kl_divs.append(sample_kl_div)
        #            p_distances.append(sample_p_distances)

        N_events += 1
        if rng.uniform() < h / (h + r):
            # external information draw event
            node = rng0.choice(nodes)
            node.set_updated_belief(
                info_in=world.get_belief_sample(llf_world, t),
                id_in=world.node_id,
                t_sys=t,
            )
        else:
            # edge event
            chatters = rng0.choice(list(G.edges()))
            sample0 = nodes[chatters[0]].get_belief_sample(llf_nodes, t)
            sample1 = nodes[chatters[1]].get_belief_sample(llf_nodes, t)
            nodes[chatters[0]].set_updated_belief(sample1, chatters[1], t)
            nodes[chatters[1]].set_updated_belief(sample0, chatters[0], t)

        t += st.expon.rvs(scale=1 / (h + r))

        # Sample post-run state
    #        if int(t / t_sample) >= sample_counter:
    #            if progress:
    #                print("Sampling at t=", t)
    #            sample_counter += 1
    #            sample_mu_nodes, sample_kl_div, sample_p_distances = ppd_func(**ppd_in)
    #            mu_nodes.append(sample_mu_nodes)
    #            kl_divs.append(sample_kl_div)
    #            p_distances.append(sample_p_distances)

    return {
        "nodes": nodes,
        "G": G,
        "world": world,
        "N_events": N_events,
        "t_end": t,
        #        "mu_nodes": mu_nodes,
        #        "kl_divs": kl_divs,
        #        "p_distances": p_distances,
        "seed": RANDOM_SEED,
    }
