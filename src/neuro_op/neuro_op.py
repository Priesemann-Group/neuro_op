import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import scipy.stats as st


RANDOM_SEED = np.random.SeedSequence().entropy
random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)


class node:
    """
    nodes with belief-holding, -sampling, and -updating behavior.

    Attributes:
    beliefs -- Numpy array of possible parameter values into which an node may hold belief
    log_probs -- Numpy array current relative log-probabilities of each belief
    likelihood -- likelihood function used during Bayesian belief updating; added for potential future extension to changing tolerance (i.e., sigma)
    diary_in -- Array of past incoming information
    diary_out -- Array of past outgoing information
    """

    def __init__(self, beliefs, log_priors, likelihood=st.norm(loc=0, scale=5)):
        """
        Initialize an node capable of updating and sampling of a world model (= beliefs & log-probabilities of each belief).
        """

        assert len(beliefs) == len(log_priors)
        self.beliefs = np.copy(beliefs)
        self.log_probs = np.copy(log_priors)
        self.likelihood = likelihood
        self.diary_in = np.array([])
        self.diary_out = np.array([])

    def set_updated_belief(self, incoming_info):
        """Bayesian update of the node's belief AND fit of new likelihood function."""

        self.diary_in = np.append(self.diary_in, incoming_info)
        self.log_probs += self.likelihood.logpdf(x=self.beliefs - incoming_info)
        self.log_probs -= np.max(self.log_probs)  # subtract max for numerical stability

    def get_belief_sample(self, size=1):
        """Sample a belief according to world model."""

        probs = logpdf_to_pdf(self.log_probs)
        sample = np.random.choice(self.beliefs, size=size, p=probs)
        self.diary_out = np.append(self.diary_out, sample)

        return sample

    def __repr__(self):
        """Return a string representation of the node."""

        return f"node(beliefs={self.beliefs}, log_probs={self.log_probs}, likelihood={self.likelihood}, diary_in={self.diary_in}, diary_out={self.diary_out})"


def build_random_network(N_nodes, N_neighbours):
    """
    Build adjacency matrix of a weighted graph of N_nodes, with random connections.
    At the start, each node has, on average, 3 connections, each connection is bidirectional and of weight 1.
    """

    p = N_neighbours / (
        N_nodes - 1
    )  # probability of connection; '-1' to exclude self-connections

    # size = (N_nodes, N_nodes)
    # network = rng.uniform(size=size)
    # network = np.where(network < p, 1, 0)
    # network = np.triu(network, k=1)   # remove self-connections
    # network2 = network + network.T
    #
    # return network2

    G = nx.gnp_random_graph(N_nodes, p).to_directed()
    return G  # nx.adjacency_matrix(G).todense()


def logpdf_to_pdf(logprobs):
    """
    Returns array of relative log probabilities as normalized relative probabilities.

    Keyword arguments:
    logprobs -- array of log probabilities
    """

    probs = np.exp(
        logprobs - np.max(logprobs)
    )  # shift logprobs before np.exp so that exp(logprobs)>0; needed due to non-normalized log-space Bayesian update

    return probs / np.sum(probs)


def dist_binning(dist, N_bins=50, range=(-20, 20)):
    """
    Returns a scipy distribution's probability mass binned into N equally-spaced bins.

    Returns normalized numpy array of scipy distribution probabilities to sample value in corresponding bin.
    Uses scipy distribution's cdf function to integrate over each bin's range.

    Keyword arguments:
    dist -- scipy distribution
    range -- interval for which binning is performed
    N_bins -- number of bins to be returned
    """

    Q_bins = np.linspace(range[0], range[1], N_bins + 1)
    Q = dist.cdf(Q_bins[1:]) - dist.cdf(Q_bins[:-1])

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
    P -- array of discrete probability distribution
    Q -- array of discrete probability distribution
    """

    terms = P * np.log(P / Q)

    # assure that KL-div is 0 for (P==0 && Q>0)
    terms[(P / Q == 0)] = 0

    return np.sum(terms)


def p_distances(mu_nodes, mu_ref, p=1, p_inv=1):
    """
    Returns an average distance between an array of nodes' inferred parameters and a reference parameter.

    Returns a distance between an array of node-inferred generative model parameters and a reference value.
    Usually, the reference value of interest is the parameter encoding/generating the real world data
    (e.g., the mean of a Gaussian distribution).
    Usually, the array of MLEs is usually created from nodes' posterior predictive distributions (PPDs).

    If p_inv = 1/p, this function returns the p-norm.
    If p_inv = 1, the return value is usually multiplied by 1/len(mu_nodes) to obtain the mean distance.
    For sum of linear distances, choose p = 1, p_inv = 1  .
    For sum of quadratic distances, choose p = 2, p_inv = 1  .

    Keyword arguments:
    mu_nodes -- array of values
    mu_ref -- reference value to which mu_nodes values are compared
    p -- value by which each difference is raised, usually 1 (linear distance) or 2 (squared distance)
    p_inv -- value by which the sum of differences is raised.
    """

    return np.sum(np.abs(mu_nodes - mu_ref) ** p) ** p_inv


def ppd_Gaussian_mu(beliefs, logprobs, N_samples=1000):
    """
    Simulate predictions using the whole posterior, with the underlying likelihood function being Gaussian.

    Posterior predictive distribution (PPD) sampling first samples paramter values of the estimand from the posterior.
    Then these sampled parameter values will be used in the likelihood function to sample predictions.
    Thereby, the PPD includes all the uncertainty (i.e., model parameter value uncertainty (from posterior) & generative uncertainty (model with given parameter values creating data stochastically).

    Keyword arguments:
    beliefs -- array of possible parameter values into which an node may hold belief
    logprobs -- array of log probabilities corresponding to 'beliefs' array
    N_samples -- number of to-be-drawn likelihood parameter values and then-sampled predictions; can in principle be split up into two separate parameters (one for parameter sampling, one for prediction sampling)
    """

    # Transform potentially non-normalized log probabilities to normalized probabilities.
    probs = logpdf_to_pdf(logprobs)

    # Sample parameter values proportional to the posterior.
    parameter_samples = np.random.choice(beliefs, p=probs, size=N_samples)

    # Generate predictions using the likelihood function.
    return st.norm.rvs(loc=parameter_samples, scale=5, size=N_samples)


def network_dynamics(nodes, G, world, h, r, t_end):
    """
    Simulate the dynamics of Graph.
    As of now, weights are constant, only beliefs change.

    Keyword arguments:
    nodes -- list of nodes, each having beliefs and nodes
    G -- networkx graph object (formerly adjacency matrix)
    world -- distribution providing stochastically blurred actual world state
    h -- rate of external information draw events
    r -- rate of edge information exchange events
    t_end -- end time of simulation
    """

    N_nodes = nx.number_of_nodes(G)
    N_edges = nx.number_of_edges(G)
    t = 0
    N_events = 0

    while t < t_end:
        dt = st.expon.rvs(scale=1 / (h + r))
        t = t + dt
        N_events += 1
        event = rng.uniform()

        if event < N_nodes * h / (N_nodes * h + N_edges * r):
            # external information draw event
            node = random.choice(nodes)
            node.set_updated_belief(world.get_belief_sample(size=1))

        else:
            # edge event
            chatters = random.choice(list(G.edges()))
            # update each node's log-probabilities with sample of edge neighbour's beliefs
            sample0 = nodes[chatters[0]].get_belief_sample(size=1)
            sample1 = nodes[chatters[1]].get_belief_sample(size=1)
            nodes[chatters[0]].set_updated_belief(sample1)
            nodes[chatters[1]].set_updated_belief(sample0)

    return (
        nodes,
        G,
        world,
        N_events,
        t_end,
    )


def run_model(
    N_nodes=100,
    N_neighbours=3,
    N_beliefs=500,
    belief_min=-50,
    belief_max=50,
    log_priors=np.zeros(500),
    likelihood=st.norm(loc=0, scale=5),
    world_dist=st.norm(loc=0, scale=5),
    h=1,
    r=1,
    t_max=1000,
):
    """
    Execute program.
    Get all parameters and initialize nodes (w. belief and log-prior distributions), network graph, and world distribution.
    Then, run simulation until t>=t_max and return simulation results.

    Keyword arguments:
    N_nodes -- number of nodes
    N_neighbours -- expected number of neighbours per node
    N_beliefs -- number of beliefs (= grid points) we consider
    belief_min -- minimum value with belief > 0
    belief_max -- maximum value with belief > 0
    world -- distribution providing stochastically blurred actual world state
    h -- world distribution sampling rate
    r -- edge neighbour's beliefs sampling rate
    t_max -- time after which simulation stops
    """

    assert N_beliefs == len(log_priors)

    beliefs = np.linspace(belief_min, belief_max, N_beliefs)
    nodes = [node(beliefs, log_priors, likelihood) for i in range(N_nodes)]
    G = build_random_network(N_nodes, N_neighbours)
    world = node(beliefs=beliefs, log_priors=world_dist.logpdf(x=beliefs))

    nodes, G, world, N_events, t_end = network_dynamics(nodes, G, world, h, r, t_max)

    return nodes, G, beliefs, world, N_events, t_end