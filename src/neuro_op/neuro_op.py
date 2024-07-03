import time

import scipy.stats as st

from .classes import *
from .measures import *
from .randomness import *
from .utils import *


def run_model_Grid(
    G,
    mu_arr,
    sd_arr,
    log_priors,
    llf_nodes,
    llf_world,
    params_world,
    h,
    r,
    t0,
    t_max,
    t_sample=1,
    sample_range=(-10, 10),
    sample_bins=101,
    sampling=False,
):
    """
    Execute program.
    Get all parameters and initialize nodes (w. belief and log-prior distributions), network graph, and world distribution.
    Then, run simulation until t>=t_max and return simulation results.

    Keyword arguments:
    G : networkx graph object
        Graph of nodes and edges
    mu_arr : np.array (floats)
        Array of mu values into which a Node may hold beliefs in.
    sd_arr : np.array (floats)
        Array of standard deviation values into which a Node may hold beliefs in.
    log_priors : list (floats)
        Prior log-probabilities of nodes
    llf_nodes : scipy.stats function
        Likelihood function (llf) of nodes
    llf_world : scipy.stats function
        Likelihood function (llf) of world
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
    progress : bool
        Whether or not to print sampling times
    """

    starttime = time.time()
    assert (len(mu_arr), len(sd_arr)) == log_priors.shape

    # Renormalize rates to keep rate per node constant
    h = h * len(G)
    r = r * len(G)

    nodesGrid = [
        NodeGrid(
            node_id=i,
            log_priors=log_priors,
        )
        for i in G.nodes()
    ]

    # Initialize world probabilities as 2D array to stay compatible with 'NodeGrid' class, by setting logprobs to -1000 (effectively zero) for all but the parameter values closest to the truth.
    world_probs = np.full((len(mu_arr), len(sd_arr)), -1000)
    mu_idx = np.argmin(np.abs(mu_arr - params_world["loc"]))
    sd_idx = np.argmin(np.abs(sd_arr - params_world["scale"]))
    world_probs[mu_idx, sd_idx] = 0
    world = NodeGrid(
        node_id=-1,
        log_priors=world_probs,
    )
    # Bin world distribution for later distance measures
    world_ppd = dist_binning(llf_world, params_world, sample_bins, sample_range)

    N_events = 0
    t = t0
    sample_counter = int(t0 / t_sample)
    mu_nodes = []
    kl_divs = []

    # Run simulation...
    while t < t_max:
        # Sample MLEs, KLDs with periodicity t_sample
        if sampling and sample_counter <= t / t_sample:
            while len(mu_nodes) <= t / t_sample - 1:
                sample_counter += 1
                mu_nodes.append(mu_nodes[-1])
                kl_divs.append(kl_divs[-1])
            sample_counter += 1
            mu_nodes.append(
                [get_MLE_grid(node.log_probs, mu_arr) for node in nodesGrid]
            )
            kl_divs.append(
                [
                    kl_divergence(
                        P=world_ppd,
                        Q=np.histogram(
                            node.get_belief_sample(
                                llf_nodes, mu_arr, sd_arr, t, ppd=True
                            ),
                            bins=sample_bins,
                            range=sample_range,
                        )[0],
                    )
                    for node in nodesGrid
                ]
            )

        # Information exchange event...
        N_events += 1
        if rng.uniform() < h / (h + r):
            # external information draw event
            node = rng0.choice(nodesGrid)
            node.set_updated_belief(
                llf_nodes,
                mu_arr,
                sd_arr,
                world.get_belief_sample(llf_world, mu_arr, sd_arr, t),
                id_in=-1,
                t_sys=t,
            )

        else:
            # edge event
            chatters = rng0.choice(list(G.edges()))
            sample0 = nodesGrid[chatters[0]].get_belief_sample(
                llf_nodes, mu_arr, sd_arr, t
            )
            sample1 = nodesGrid[chatters[1]].get_belief_sample(
                llf_nodes, mu_arr, sd_arr, t
            )
            nodesGrid[chatters[0]].set_updated_belief(
                llf_nodes, mu_arr, sd_arr, sample1, chatters[1], t
            )
            nodesGrid[chatters[1]].set_updated_belief(
                llf_nodes, mu_arr, sd_arr, sample0, chatters[0], t
            )

        t += st.expon.rvs(scale=1 / (h + r))

    # Sample post-execution system PPDs, distance measures (KL-div, p-distance), if skipped in last iteration
    if sampling and sample_counter <= t / t_sample:
        while len(mu_nodes) <= t / t_sample - 1:
            sample_counter += 1
            mu_nodes.append(mu_nodes[-1])
            kl_divs.append(kl_divs[-1])
        sample_counter += 1
        mu_nodes.append([get_MLE_grid(node.log_probs, mu_arr) for node in nodesGrid])
        kl_divs.append(
            [
                kl_divergence(
                    P=world_ppd,
                    Q=np.histogram(
                        node.get_belief_sample(llf_nodes, mu_arr, sd_arr, t, ppd=True),
                        bins=sample_bins,
                        range=sample_range,
                    )[0],
                )
                for node in nodesGrid
            ]
        )

    dict_out = {
        "nodesGrid": nodesGrid,
        "world": world,
        "G": G,
        "N_events": N_events,
        "t_end": t,
        "t_start": time.strftime("%Y-%m-%d--%H-%M", time.localtime(starttime)),
        "t_exec": time.time() - starttime,
        "seed": RANDOM_SEED,
        "mu_arr": mu_arr,
        "sd_arr": sd_arr,
    }
    if sampling:
        dict_out["mu_nodes"] = mu_nodes
        dict_out["kl_divs"] = kl_divs

    return dict_out


def run_model_singleGrid(
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
    t_sample,
    sample_bins,
    sample_range,
    p_distance_params,
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

    nodesNormal = [
        NodeNormal(
            node_id=i,
            log_priors=log_priors,
            params_node=params_node,
        )
        for i in G.nodes()
    ]
    world = NodeNormal(
        node_id=-1,
        log_priors=llf_world.logpdf(**params_world, x=beliefs),
        params_node=params_world,
    )
    ppd_func = ppd_distances_Gaussian
    ppd_in = dict(
        llf_nodes=llf_nodes,
        llf_world=llf_world,
        beliefs=beliefs,
        nodes=nodesNormal,
        world=world,
        sample_bins=sample_bins,
        sample_range=sample_range,
        p_distance_params=p_distance_params,
    )

    # Run simulation...
    N_events = 0
    t = t0
    sample_counter = int(t0 / t_sample)
    mu_nodes = []
    kl_divs = []
    p_distances = []

    while t < t_max:
        # Sample system PPDs, distance measures (KL-div, p-distance) with periodicity t_sample
        if int(t / t_sample) >= sample_counter:
            if progress:
                print("Sampling at t=", t, "\t, aka", (t / t_max), "\t of runtime.")
            sample_counter += 1
            sample_mu_nodes, sample_kl_div, sample_p_distances = ppd_func(**ppd_in)
            mu_nodes.append(sample_mu_nodes)
            kl_divs.append(sample_kl_div)
            p_distances.append(sample_p_distances)

        # Information exchange event...
        N_events += 1
        if rng.uniform() < h / (h + r):
            # external information draw event
            node = rng0.choice(nodesNormal)
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
            sample0 = nodesNormal[chatters[0]].get_belief_sample(
                beliefs=beliefs, t_sys=t
            )
            sample1 = nodesNormal[chatters[1]].get_belief_sample(
                beliefs=beliefs, t_sys=t
            )
            nodesNormal[chatters[0]].set_updated_belief(
                llf_nodes, beliefs, sample1, chatters[1], t
            )
            nodesNormal[chatters[1]].set_updated_belief(
                llf_nodes, beliefs, sample0, chatters[0], t
            )

        t += st.expon.rvs(scale=1 / (h + r))

    # Sample post-execution system PPDs, distance measures (KL-div, p-distance), if skipped in last iteration
    if int(t / t_sample) >= sample_counter:
        if progress:
            print("Sampling at t=", t)
        sample_counter += 1
        sample_mu_nodes, sample_kl_div, sample_p_distances = ppd_func(**ppd_in)
        mu_nodes.append(sample_mu_nodes)
        kl_divs.append(sample_kl_div)
        p_distances.append(sample_p_distances)

    dict_out = {
        "nodesNormal": nodesNormal,
        "world": world,
        "G": G,
        "N_events": N_events,
        "t_end": t,
        "t_start": time.strftime("%Y-%m-%d--%H-%M", time.localtime(starttime)),
        "t_exec": time.time() - starttime,
        "seed": RANDOM_SEED,
        "beliefs": beliefs,
        "mu_nodes": mu_nodes,
        "kl_divs": kl_divs,
        "p_distances": p_distances,
    }

    return dict_out


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
    t_sample,
    sample_bins,
    sample_range,
    sampling,
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
    """

    starttime = time.time()
    # Renormalize rates to keep rate per node constant
    h = h * len(G)
    r = r * len(G)
    # Set up simulation environment (nodes, world, sampling function/inputs)
    nodesConj = [
        NodeConjMu(node_id=i, params_node=params_node, sd_llf=sd_llf) for i in G.nodes()
    ]
    world = NodeConjMu(node_id=-1, params_node=params_world)
    if sampling:
        world_binned = dist_binning(llf_world, params_world, sample_bins, sample_range)
    N_events = 0
    t = t0
    sample_counter = int(t0 / t_sample)
    mu_nodes = []
    kl_divs = []

    # Run simulation...
    while t < t_max:
        # Sample MLEs, KLDs with periodicity t_sample
        if sampling and sample_counter <= t / t_sample:
            while len(mu_nodes) <= t / t_sample - 1:
                sample_counter += 1
                mu_nodes.append(mu_nodes[-1])
                kl_divs.append(kl_divs[-1])
            sample_counter += 1
            mu_nodes.append([node.params_node["loc"] for node in nodesConj])
            kl_divs.append(
                [
                    kl_divergence(
                        P=world_binned,
                        Q=dist_binning(
                            llf_nodes,
                            {
                                "loc": node.params_node["loc"],
                                "scale": node.params_node["scale"] + node.sd_llf,
                            },
                            sample_bins,
                            sample_range,
                        ),
                    )
                    for node in nodesConj
                ]
            )

        N_events += 1
        if rng.uniform() < h / (h + r):
            # external information draw event
            node = rng0.choice(nodesConj)
            node.set_updated_belief(
                info_in=world.get_belief_sample(llf_world, t),
                id_in=world.node_id,
                t_sys=t,
            )
        else:
            # edge event
            chatters = rng0.choice(list(G.edges()))
            sample0 = nodesConj[chatters[0]].get_belief_sample(llf_nodes, t)
            sample1 = nodesConj[chatters[1]].get_belief_sample(llf_nodes, t)
            nodesConj[chatters[0]].set_updated_belief(sample1, chatters[1], t)
            nodesConj[chatters[1]].set_updated_belief(sample0, chatters[0], t)

        t += st.expon.rvs(scale=1 / (h + r))

    # Sample post-run state
    if sampling and sample_counter <= t / t_sample:
        while len(mu_nodes) <= t / t_sample - 1:
            sample_counter += 1
            mu_nodes.append(mu_nodes[-1])
            kl_divs.append(kl_divs[-1])
        sample_counter += 1
        mu_nodes.append([node.params_node["loc"] for node in nodesConj])
        kl_divs.append(
            [
                kl_divergence(
                    P=world_binned,
                    Q=dist_binning(
                        llf_nodes,
                        {
                            "loc": node.params_node["loc"],
                            "scale": node.params_node["scale"] + node.sd_llf,
                        },
                        sample_bins,
                        sample_range,
                    ),
                )
                for node in nodesConj
            ]
        )

    dict_out = {
        "nodesConj": nodesConj,
        "world": world,
        "G": G,
        "N_events": N_events,
        "t_end": t,
        "t_start": time.strftime("%Y-%m-%d--%H-%M", time.localtime(starttime)),
        "t_exec": time.time() - starttime,
        "seed": RANDOM_SEED,
    }
    if sampling:
        dict_out["mu_nodes"] = mu_nodes
        dict_out["kl_divs"] = kl_divs

    return dict_out
