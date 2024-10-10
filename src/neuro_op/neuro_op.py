import time

import scipy.stats as st

from .classes import *
from .measures import *
from .randomness import *
from .utils import *


def run_Grid(
    G,
    llf_nodes,
    mu_arr,
    sd_arr,
    log_priors,
    llf_world,
    params_world,
    h,
    r,
    t0,
    t_max,
    t_sample,
    sample_range,
    sample_bins,
    sampling,
    init_rngs,
    seed,
):
    """
    Run network dynamics with nodesGrid class & return results.
    """

    starttime = time.time()
    assert (len(mu_arr), len(sd_arr)) == log_priors.shape
    # Set up simulation environment (nodes, world, utility variables)...
    # For reproducibility, (re)initialize RNGs with current seed
    if init_rngs:
        global RANDOM_SEED, rng0, rng
        if bool(int(seed)):
            RANDOM_SEED = seed
        RANDOM_SEED, rng0, rng = init_seeds(RANDOM_SEED)
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
    world_probs = np.full((len(mu_arr), len(sd_arr)), -1000)
    mu_idx = np.argmin(np.abs(mu_arr - params_world["loc"]))
    sd_idx = np.argmin(np.abs(sd_arr - params_world["scale"]))
    world_probs[mu_idx, sd_idx] = 0
    world = NodeGrid(
        node_id=-1,
        log_priors=world_probs,
    )
    world_ppd = dist_binning(llf_world, params_world, sample_bins, sample_range)
    N_events = 0
    t = t0
    sample_counter = int(t0 / t_sample)
    mu_nodes = []
    kl_divs = []

    # Run simulation...
    while t < t_max:
        # Sample MLEs, KLDs with periodicity t_sample...
        if sampling and sample_counter <= t / t_sample:
            while len(mu_nodes) <= t / t_sample - 1:
                sample_counter += 1
                mu_nodes.append(mu_nodes[-1])
                kl_divs.append(kl_divs[-1])
            sample_counter += 1
            mu_nodes.append(
                [get_MLE_Grid(node.log_probs, mu_arr) for node in nodesGrid]
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
            # event: world shares information
            node = rng0.choice(nodesGrid)
            node.set_updated_belief(
                llf_nodes,
                mu_arr,
                sd_arr,
                world.get_belief_sample(llf_world, mu_arr, sd_arr, t),
                id_in=world.node_id,
                t_sys=t,
            )
        else:
            # event: two neighbours share information
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

    # Post-run sampling...
    if sampling and sample_counter <= t / t_sample:
        while len(mu_nodes) <= t / t_sample - 1:
            sample_counter += 1
            mu_nodes.append(mu_nodes[-1])
            kl_divs.append(kl_divs[-1])
        sample_counter += 1
        mu_nodes.append([get_MLE_Grid(node.log_probs, mu_arr) for node in nodesGrid])
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

    # Return results...
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


def run_GridMu(
    G,
    llf_nodes,
    mu_arr,
    log_priors,
    sd_llf,
    llf_world,
    mu_world,
    sd_world,
    h,
    r,
    t0,
    t_max,
    t_sample,
    sample_range,
    sample_bins,
    sampling,
    init_rngs,
    seed,
):
    """
    Run network dynamics with NodeGridMu class & return results.
    """
    starttime = time.time()
    assert len(mu_arr) == len(log_priors)
    # Set up simulation environment (nodes, world, utility variables)...
    # For reproducibility, (re)initialize RNGs with current seed
    if init_rngs:
        global RANDOM_SEED, rng0, rng
        if bool(int(seed)):
            RANDOM_SEED = seed
        RANDOM_SEED, rng0, rng = init_seeds(RANDOM_SEED)
    # Renormalize rates to keep rate per node constant
    h = h * len(G)
    r = r * len(G)
    nodesGridMu = [
        NodeGridMu(
            node_id=i,
            log_priors=log_priors,
        )
        for i in G.nodes()
    ]
    world = NodeGridMu(
        node_id=-1,
        log_priors=llf_world.logpdf(loc=mu_world, scale=sd_world, x=mu_arr),
    )
    world_ppd = dist_binning(
        llf_world, {"loc": mu_world, "scale": sd_world}, sample_bins, sample_range
    )
    N_events = 0
    t = t0
    sample_counter = int(t0 / t_sample)
    mu_nodes = []
    kl_divs = []

    # Run simulation...
    while t < t_max:
        # Sample MLEs, KLDs with periodicity t_sample
        if sampling and sample_counter <= t / t_sample:
            # If delta_t > t_sample, use latest sample as for skipped sampling points
            while len(mu_nodes) <= t / t_sample - 1:
                sample_counter += 1
                mu_nodes.append(mu_nodes[-1])
                kl_divs.append(kl_divs[-1])
            sample_counter += 1
            mu_nodes.append(
                [get_MLE_Grid(node.log_probs, mu_arr) for node in nodesGridMu]
            )
            kl_divs.append(
                [
                    kl_divergence(
                        P=world_ppd,
                        Q=np.histogram(
                            node.get_belief_sample(
                                llf_nodes, mu_arr, sd_llf, t, ppd=True
                            ),
                            bins=sample_bins,
                            range=sample_range,
                        )[0],
                    )
                    for node in nodesGridMu
                ]
            )
        # Information exchange event...
        N_events += 1
        if rng.uniform() < h / (h + r):
            # event: world shares information
            node = rng0.choice(nodesGridMu)
            node.set_updated_belief(
                llf_nodes,
                mu_arr,
                sd_llf,
                info_in=world.get_belief_sample(llf_world, mu_arr, sd_llf, t),
                id_in=world.node_id,
                t_sys=t,
            )
        else:
            # event: two neighbours share information
            chatters = rng0.choice(list(G.edges()))
            sample0 = nodesGridMu[chatters[0]].get_belief_sample(
                llf_nodes, mu_arr, sd_llf, t
            )
            sample1 = nodesGridMu[chatters[1]].get_belief_sample(
                llf_nodes, mu_arr, sd_llf, t
            )
            nodesGridMu[chatters[0]].set_updated_belief(
                llf_nodes, mu_arr, sd_llf, sample1, chatters[1], t
            )
            nodesGridMu[chatters[1]].set_updated_belief(
                llf_nodes, mu_arr, sd_llf, sample0, chatters[0], t
            )
        t += st.expon.rvs(scale=1 / (h + r))

    # Post-run sampling...
    if sampling and sample_counter <= t / t_sample:
        # If delta_t > t_sample, use latest sample as for skipped sampling points
        while len(mu_nodes) <= t / t_sample - 1:
            sample_counter += 1
            mu_nodes.append(mu_nodes[-1])
            kl_divs.append(kl_divs[-1])
        sample_counter += 1
        mu_nodes.append([get_MLE_Grid(node.log_probs, mu_arr) for node in nodesGridMu])
        kl_divs.append(
            [
                kl_divergence(
                    P=world_ppd,
                    Q=np.histogram(
                        node.get_belief_sample(llf_nodes, mu_arr, sd_llf, t, ppd=True),
                        bins=sample_bins,
                        range=sample_range,
                    )[0],
                )
                for node in nodesGridMu
            ]
        )

    # Return results...
    dict_out = {
        "nodesGridMu": nodesGridMu,
        "world": world,
        "G": G,
        "N_events": N_events,
        "t_end": t,
        "t_start": time.strftime("%Y-%m-%d--%H-%M", time.localtime(starttime)),
        "t_exec": time.time() - starttime,
        "seed": RANDOM_SEED,
        "mu_arr": mu_arr,
    }
    if sampling:
        dict_out["mu_nodes"] = mu_nodes
        dict_out["kl_divs"] = kl_divs

    return dict_out


def run_ConjMu(
    G,
    llf_nodes,
    params_node,
    sd_llf,
    llf_world,
    params_world,
    h,
    r,
    t0,
    t_max,
    t_sample,
    sampling,
    init_rngs,
    seed,
):
    """
    Run network dynamics with nodesConjMu class & return results.
    """

    starttime = time.time()
    # Set up simulation environment (nodes, world, utility variables)...
    # For reproducibility, (re)initialize RNGs with current or supplied seed
    if init_rngs:
        global RANDOM_SEED, rng0, rng
        if bool(int(seed)):
            RANDOM_SEED = seed
        RANDOM_SEED, rng0, rng = init_seeds(RANDOM_SEED)
    nodesConjMu = [
        NodeConjMu(node_id=i, params_node=params_node, sd_llf=sd_llf) for i in G.nodes()
    ]
    world = NodeConjMu(node_id=-1, params_node=params_world)
    t = t0
    sample_counter = int(t0 / t_sample)
    mu_nodes = []
    sd_nodes = []
    kl_divs = []
    N_events = 0

    # Run simulation...
    while t < t_max:
        # Sample MLEs, KLDs with periodicity t_sample...
        if sampling and sample_counter <= t / t_sample:
            while (
                len(mu_nodes) <= t / t_sample - 1
            ):  # If delta_t > t_sample, use latest sample as for skipped sampling points
                sample_counter += 1
                mu_nodes.append(mu_nodes[-1])
                sd_nodes.append(sd_nodes[-1])
                kl_divs.append(kl_divs[-1])
            sample_counter += 1
            mu_q = np.array([node.params_node["loc"] for node in nodesConjMu])
            sd_q = np.array(
                [node.params_node["scale"] + node.sd_llf for node in nodesConjMu]
            )
            mu_p = world.params_node["loc"]
            sd_p = world.params_node["scale"]
            mu_nodes.append(mu_q.tolist())
            sd_nodes.append(sd_q.tolist())
            kl_divs.append(
                (
                    np.log(sd_q / sd_p)
                    + (sd_p**2 + (mu_p - mu_q) ** 2) / (2 * sd_q**2)
                    - 0.5
                ).tolist()
            )
        # Information exchange event...
        N_events += 1
        if rng.uniform() < h / (h + r):
            # event: world shares information
            node = rng0.choice(nodesConjMu)
            node.set_updated_belief(
                info_in=world.get_belief_sample(llf_world, t),
                id_in=world.node_id,
                t_sys=t,
            )
        else:
            # event: two neighbours share information
            chatters = rng0.choice(list(G.edges()))
            sample0 = nodesConjMu[chatters[0]].get_belief_sample(llf_nodes, t)
            sample1 = nodesConjMu[chatters[1]].get_belief_sample(llf_nodes, t)
            nodesConjMu[chatters[0]].set_updated_belief(sample1, chatters[1], t)
            nodesConjMu[chatters[1]].set_updated_belief(sample0, chatters[0], t)
        t += st.expon.rvs(scale=1 / (len(G) * (h + r)))

    # Post-run sampling...
    if sampling and sample_counter <= t / t_sample:
        while len(mu_nodes) <= t / t_sample - 1:
            sample_counter += 1
            mu_nodes.append(mu_nodes[-1])
            sd_nodes.append(sd_nodes[-1])
            kl_divs.append(kl_divs[-1])
        sample_counter += 1
        mu_q = np.array([node.params_node["loc"] for node in nodesConjMu])
        sd_q = np.array(
            [node.params_node["scale"] + node.sd_llf for node in nodesConjMu]
        )
        mu_p = world.params_node["loc"]
        sd_p = world.params_node["scale"]
        mu_nodes.append(mu_q.tolist())
        sd_nodes.append(sd_q.tolist())
        kl_divs.append(
            (
                np.log(sd_q / sd_p)
                + (sd_p**2 + (mu_p - mu_q) ** 2) / (2 * sd_q**2)
                - 0.5
            ).tolist()
        )

    # Return results...
    dict_out = {
        "nodesConjMu": nodesConjMu,
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
        dict_out["sd_nodes"] = sd_nodes
        dict_out["kl_divs"] = kl_divs

    return dict_out
