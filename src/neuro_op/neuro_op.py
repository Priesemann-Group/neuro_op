# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import scipy.stats as st


# Initialize randomness and RNGs
RANDOM_SEED = np.random.SeedSequence().entropy
random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)


class NodeNormal:
    """
    Nodes with grid-wise belief-holding, -sampling, and -updating behavior.

    Attributes:
    node_id : int
        Supposedly unique identifier of the node
    log_probs : array
        Unnormalized log-probabilities, corresponding to external 'beliefs' array
    llh_params : dict
        Necessary paramters for the node's model (==likelihood function p(data|parameters) )
    diary_in : list
        Past incoming information & metadata, structured [ [info_in_0, id_in_0, t_sys_0], [info_in_1, id_in_1, t_sys_1], ... ]
    diary_out : list
        Shared own information & metadata, structured [ [info_out_0, t_sys_0], [info_out_1, t_sys_1], ... ]
    """

    def __init__(
        self,
        node_id,
        log_priors,
        llf_params=dict(  # Parameters defining the likelihood function (llf), normal distribution by default
            mu=0, sigma=5
        ),
        diary_in=[],
        diary_out=[],
    ):
        """
        Initialize a Node capable of updating and sampling of a grid-stored world model (= beliefs & log-probabilities of each belief).
        """

        self.node_id = node_id
        self.log_probs = np.copy(log_priors)
        self.llf_params = llf_params
        self.diary_in = diary_in.copy()
        self.diary_out = diary_out.copy()

    def set_updated_belief(self, beliefs, info_in, id_in, t_sys):
        """Bayesian update of the Node's belief, based on incoming info 'info_in' from node with id 'id_in' at system time 't_sys'."""

        self.diary_in += [[info_in, id_in, t_sys]]
        log_llf = st.norm(
            loc=self.llf_params["mu"], scale=self.llf_params["sigma"]
        ).logpdf
        self.log_probs += log_llf(x=beliefs - info_in)
        self.log_probs -= np.max(self.log_probs)  # subtract max for numerical stability

    def get_belief_sample(self, beliefs, t_sys):
        """
        Sample a belief "mu == <belief value>', proportional to relative plausabilities 'probs'.
        """

        probs = logpdf_to_pdf(self.log_probs)
        info_out = rng.choice(beliefs, p=probs)
        self.diary_out += [[info_out, t_sys]]
        return info_out


# class LaplaceNode:
#    """
#    Nodes with Laplace-approximated belief-holding, -sampling, and -updating behavior.
#    """
#
#
#    def __init__(
#        self,
#        node_id,
#        mu_init=0,  # Prior mean
#        sigma_init=50,  # Prior standard deviation
#        diary_in=[],
#        diary_out=[],
#    ):
#        """
#        Initialize a Node capable of updating and sampling of a parameterized world model (= beliefs & log-probabilities of each belief).
#        """
#
#        self.node_id = node_id
#        self.mu = mu_init
#        self.sigma = sigma_init
#        self.llh = st.norm(loc=self.mu, scale=self.sigma)
#        self.diary_in = diary_in.copy()
#        self.diary_out = diary_out.copy()
#
#    def set_updated_belief(self, info_in, id_in, t_sys):
#        """Naive Bayesian-like (parameterized) belief update."""
#        self.diary_in += [[info_in, id_in, t_sys]]
#        sigma_data = np.sqrt(np.array(self.diary_in)[:, 0].var())
#        self.mu = (self.mu * sigma_data**2 + info_in * self.sigma**2) / (
#            sigma_data**2 + self.sigma**2
#        )
#        self.sigma = 1 / (1 / self.sigma**2 + 1 / sigma_data**2)
#        self.llh = st.norm(loc=self.mu, scale=self.sigma)
#
#    def get_belief_sample(self, t_sys, size=1):
#        """
#        Sample beliefs proportional to relative plausabilities.
#
#        Returns a list of 'int(size)' times [sample, node_id, t_sys].
#        """
#
#        sample = [
#            [i, self.node_id, t_sys]
#            for i in st.norm(loc=self.mu, scale=self.sigma).rvs(size=size)
#        ]
#        self.diary_out += [sample]
#
#        return sample
#


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
    Build and return a directed stochastic block model graph.

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


def dist_binning(logpdf, N_bins=50, range=(-20, 20)):
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
    Q = np.exp(logpdf(Q_bins[:-1])) + np.exp(logpdf(Q_bins[1:]))

    return Q / np.sum(Q)


def kl_divergence(P, Q):
    """
    Returns Kullback-Leibler divergence between two identically binned discrete probability distributions.

    Returns Kullback-Leibler divergence in base b=e (i.e., nats).
    Usually, P is the to-be-tested (and/or data-based) distribution and Q the reference distribution.

    If you want to use samples as basis of your inputs, you can normalize and bin them via
    > inputs = np.histogram(samples, bins=N_bins, range=(sample_space_min, sample_space_max)
    > inputs = inputs[0] / np.sum(inputs[0])

    Keyword arguments:
    P : iterable
        recorded discrete probability distribution
    Q : iterable
        reference discrete probability distribution
    """

    # Normalize P and Q
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # kind of wackily add machine epsilon to all elements of Q to avoid 'divided by zero' errors
    epsilon = 7.0 / 3 - 4.0 / 3 - 1
    Q += epsilon
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
    p : float
        value by which each difference is raised, usually 1 (linear distance) or 2 (squared distance)
    p_inv : float
        value by which the sum of differences is raised, usually 1 (mean) or 1/p (p-norm)
    """

    return np.sum(np.abs(param_node - param_ref) ** p) ** p_inv


def ppd_Gaussian_mu(beliefs, logprobs, sigma, N_samples=1000):
    """
    Simulate predictions using the whole posterior, with the underlying likelihood logprobability funtion (llh) being Gaussian.

    Posterior predictive distribution (PPD) sampling first samples paramter values of the estimand from the posterior.
    Then these sampled parameter values will be used in llh  to sample predictions.
    Thereby, the PPD includes all the uncertainty (i.e., model parameter value uncertainty (from posterior) & generative uncertainty (model with given parameter values creating data stochastically).

    Keyword arguments:
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

    # Sample parameter values proportional to the posterior.
    mu_samples = np.random.choice(beliefs, p=probs, size=N_samples)

    # Generate predictions using the llh method.
    return st.norm.rvs(loc=mu_samples, scale=sigma)


def ppd_distances_Gaussian(
    beliefs,
    nodes,
    world,
    sample_bins=50,
    sample_range=(-20, 20),
    p_distances_params=[],
    world_out=False,
):
    """
    Return approximated distances between system nodes' PPDs and world state's distribution and binning used during approximation.

    Approximates distance between system nodes' posterior predictive distributions (PPDs) and world state's distribution.
    PPDs and world state distributions are approximated by histogram-binning posterior predictive samples of each distribution.
    Then, the wanted distance (KL divergence or p-distance) is calculated between each node distribution and the world distribution.

    Keyword arguments:
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
            beliefs, node.log_probs, node.llf_params["sigma"], N_samples=1000
        )
        for node in nodes
    ]

    ppds = [
        np.histogram(i, bins=sample_bins, range=sample_range)[0] for i in ppd_samples
    ]  # create PPD approximations via sampling and binning into histograms

    if world.diary_out:
        ppd_world_out = np.histogram(  # world PPD from all information shared to the network. Also stores binning used for all PPDs.
            np.array(world.diary_out)[:, 0], bins=sample_bins, range=sample_range
        )
        ppd_world_out = ppd_world_out[0] / np.sum(
            ppd_world_out[0]
        )  # normalize world_out PPD
    else:
        ppd_world_out = np.zeros(sample_bins)

    ppd_world_true = dist_binning(
        st.norm(loc=world.llf_params["mu"], scale=world.llf_params["sigma"]).logpdf,
        sample_bins,
        sample_range,
    )

    # Get MLEs of each node's PPD -- note this implementation is not robust to PPDs with multiple peaks of same height
    argmax = [np.where(i == np.max(i))[0] for i in ppds]
    argmax = [i[len(i) // 2] for i in argmax]
    mu_nodes = [(ppd_bins[i] + ppd_bins[i + 1]) / 2 for i in argmax]

    # Get KL-divergences of each node's PPD
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
    if p_distances_params:
        # First approach: Go for MLE comparisons
        argmax = np.argmax(ppd_world_out)
        mu_world_out = (ppd_bins[argmax] + ppd_bins[argmax + 1]) / 2
        argmax = np.argmax(ppd_world_true)
        mu_world_true = (ppd_bins[argmax] + ppd_bins[argmax + 1]) / 2

    p_distances = []
    for p in p_distances_params:
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


def network_dynamics(
    nodes,
    world,
    G,
    beliefs,
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
    Simulate the dynamics of Graph and sample distances between world state and node's beliefs via PPD comparisons.
    As of now, weights are constant, only beliefs change.

    Runtime order of magnitude ~ 1s/1000 events (on Dell XPS 13 9370)

    Keyword arguments:
    nodes : list of Node objects
        Nodes inferring the world state
    world : Node object
        Node representing the world state
    G : networkx graph object
        Graph constraining node interactions
    beliefs : iterable
        Possible parameter values into which a node may hold belief
    h : float
        Graph size normalized rate of external information draw events
    r : float
        Graph size normalized rate of edge information exchange events
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
            sample_mu_nodes, sample_kl_div, sample_p_distances = ppd_distances_Gaussian(
                beliefs,
                nodes,
                world,
                sample_bins,
                sample_range,
                p_distance_params,
            )
            mu_nodes.append(sample_mu_nodes)
            kl_divs.append(sample_kl_div)
            p_distances.append(sample_p_distances)

        N_events += 1
        if rng.uniform() < h / (h + r):
            # external information draw event
            node = random.choice(nodes)
            node.set_updated_belief(
                beliefs=beliefs,
                info_in=world.get_belief_sample(beliefs, t),
                id_in=world.node_id,
                t_sys=t,
            )
        else:
            # edge event
            chatters = random.choice(list(G.edges()))
            # update each node's log-probabilities with sample of edge neighbour's beliefs
            sample0 = nodes[chatters[0]].get_belief_sample(beliefs=beliefs, t_sys=t)
            sample1 = nodes[chatters[1]].get_belief_sample(beliefs=beliefs, t_sys=t)
            nodes[chatters[0]].set_updated_belief(beliefs, sample1, chatters[1], t)
            nodes[chatters[1]].set_updated_belief(beliefs, sample0, chatters[0], t)

        dt = st.expon.rvs(scale=1 / (h + r))
        t = t + dt

    # Sample post-execution system PPDs, distance measures (KL-div, p-distance), if skipped in last iteration
    if int(t / t_sample) >= sample_counter:
        if progress:
            print("Sampling at t=", t)
        sample_counter += 1
        sample_mu_nodes, sample_kl_div, sample_p_distances = ppd_distances_Gaussian(
            beliefs,
            nodes,
            world,
            sample_bins,
            sample_range,
            p_distance_params,
        )
        mu_nodes.append(sample_mu_nodes)
        kl_divs.append(sample_kl_div)
        p_distances.append(sample_p_distances)

    return (
        nodes,
        world,
        G,
        N_events,
        t,
        mu_nodes,
        kl_divs,
        p_distances,
    )


def run_model_Normal(
    G,
    beliefs,
    llf_params,
    world_params,
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
    laplace,
):
    """
    Execute program.
    Get all parameters and initialize nodes (w. belief and log-prior distributions), network graph, and world distribution.
    Then, run simulation until t>=t_max and return simulation results.

    Keyword arguments:
    G : networkx graph object
        Graph of nodes and edges
    beliefs : list (floats)
        Possible parameter values into which a Node may hold beliefs in
    llf_params : dict
        Parameters defining the likelihood function (llf) of nodes, concerning a Gaussian by default
    world_params : dict
        Parameters defining the likelihood function (llf) of the world, concerning a Gaussian by default
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
    laplace : bool
        Whether or not to use Laplace-approximated nodes


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

    assert len(beliefs) == len(log_priors)

    if laplace:
        nodes = [LaplaceNode(node_id=i) for i in range(len(G))]
        world = LaplaceNode(node_id=-1)
    else:
        nodes = [
            NodeNormal(
                node_id=i,
                log_priors=log_priors,
                llf_params=llf_params,
            )
            for i in range(len(G))
        ]
        world = NodeNormal(
            node_id=-1,
            log_priors=st.norm(
                loc=world_params["mu"], scale=world_params["sigma"]
            ).logpdf(beliefs),
            llf_params=world_params,
        )

    # Renormalize rates to keep rate per node constant
    h = h * len(G)
    r = r * len(G)

    nodes, world, G, N_events, t_end, mu_nodes, kl_divs, p_distances = network_dynamics(
        nodes,
        world,
        G,
        beliefs,
        h,
        r,
        t0,
        t_max,
        t_sample,
        sample_bins,
        sample_range,
        p_distance_params,
        progress,
    )

    return {
        "nodes": nodes,
        "G": G,
        "beliefs": beliefs,
        "world": world,
        "N_events": N_events,
        "t_end": t_end,
        "mu_nodes": mu_nodes,
        "kl_divs": kl_divs,
        "p_distances": p_distances,
        "seed": RANDOM_SEED,
    }


# Reference input for 'run_model' function. For description of contents, see 'run_model' function docstring.
input_standard = dict(
    G=build_random_network(N_nodes=100, N_neighbours=11),  # networkx graph object
    beliefs=np.linspace(  # beliefs considered by each node
        start=-50,  # min. considered belief value
        stop=50,  # max. considered belief value
        num=500,  # number of considered belief values
    ),
    llf_params=dict(  # Likelihood function (llf) parameters of nodes, Gaussian by default
        mu=0,
        sigma=5,
    ),
    world_params=dict(  # Likelihood function (llf) parameters of to-be-approximated world state, Gaussian by default
        mu=0,
        sigma=5,
    ),
    log_priors=np.zeros(500),  # Prior log-probabilities of nodes
    # Dynamics parameters (rates, simulation times)
    h=1,
    r=1,
    t0=0,
    t_max=100,
    # Sampling parameters
    t_sample=2,
    sample_bins=50,
    sample_range=(-20, 20),
    p_distance_params=[(1, 1), (2, 1)],
    # Switches...
    progress=False,
    laplace=False,
)
