# Classes used in project. Mainly concerns network nodes.
import numpy as np
import scipy.stats as st

from .randomness import rng
from .utils import logpdf_to_pdf


class NodeGrid:
    """
    Nodes with grid-wise belief-holding, -sampling, and -updating behavior.

    Attributes:
    node_id : int
        Supposedly unique identifier of the node
    log_probs : array
        Unnormalized log-probabilities, corresponding to external 'beliefs' array
    params_node : dict
        Likelihood function paramters of node's model
    diary_in : list
        Past incoming information & metadata, structured [ [info_in_0, id_in_0, t_sys_0], [info_in_1, id_in_1, t_sys_1], ... ]
    diary_out : list
        Shared own information & metadata, structured [ [info_out_0, t_sys_0], [info_out_1, t_sys_1], ... ]
    """

    def __init__(
        self,
        node_id,
        log_priors,
        diary_in=[],
        diary_out=[],
    ):
        """
        Initialize a Node capable of updating and sampling of a grid-stored world model (= beliefs & log-probabilities of each belief).
        """
        self.node_id = node_id
        self.log_probs = np.copy(log_priors)
        self.diary_in = diary_in.copy()
        self.diary_out = diary_out.copy()

    def set_updated_belief(self, llf, mu_arr, sd_arr, info_in, id_in, t_sys):
        """Bayesian update of the Node's belief, based on incoming info 'info_in' from node with id 'id_in' at system time 't_sys'."""
        self.diary_in += [[info_in, id_in, t_sys]]
        for i, mu in enumerate(mu_arr):
            self.log_probs[i] += llf.logpdf(x=info_in, loc=mu, scale=sd_arr)
        self.log_probs -= np.max(self.log_probs)  # subtract max for numerical stability

    def get_belief_sample(self, llf, mu_arr, sd_arr, t_sys, ppd=False, N_ppd=1000):
        """
        Return ppd samples "data=...", i.e. with {mu, sd} chosen proportionally to 'log_probs'.
        """
        if ppd:
            size = N_ppd
        else:
            size = 1
        flat_probs = logpdf_to_pdf(self.log_probs).flatten()
        mu_idx, sd_idx = np.unravel_index(
            rng.choice(len(flat_probs), p=flat_probs, size=size), self.log_probs.shape
        )
        info_out = llf.rvs(
            loc=mu_arr[mu_idx],
            scale=sd_arr[sd_idx],
        )
        if not ppd:
            self.diary_out += [[info_out, t_sys]]
        return info_out


class NodeGridMu:
    """
    Nodes with grid-wise belief-holding, -sampling, and -updating behavior for mean mu, but standard deviation convergence as if given by Normal conjugate llf & prior distributions.
    """

    def __init__(
        self,
        node_id,
        log_priors,  # Relative plausabilities of different mean values
        diary_in=[],
        diary_out=[],
    ):
        """
        Initialize a Node capable of updating and sampling of a grid-stored world model "mu ~ log_probs".
        """
        self.node_id = node_id
        self.log_probs = np.copy(log_priors)
        self.diary_in = diary_in.copy()
        self.diary_out = diary_out.copy()

    def set_updated_belief(self, llf_nodes, mu_arr, sd_llf, info_in, id_in, t_sys):
        """Grid-wise update of mean belief, closed-form update of standard deviation; assuming Normal llf with known standard deviation."""
        self.diary_in += [[info_in, id_in, t_sys]]
        self.log_probs += llf_nodes.logpdf(
            x=info_in, loc=mu_arr, scale=sd_llf
        )  # Bayesian update (log_post ~ log_prior + log_llh)
        self.log_probs -= np.max(self.log_probs)  # subtract max for numerical stability

    def get_belief_sample(self, llf, mu_arr, sd_llf, t_sys, ppd=False, N_ppd=1000):
        """
        Return ppd samples "data=...", llf samples with mu chosen proportionally to 'log_probs.
        """
        if ppd:
            size = N_ppd
        else:
            size = 1
        mu_sample = mu_arr[
            rng.choice(
                len(self.log_probs),
                p=logpdf_to_pdf(self.log_probs),
                size=size,
            )
        ]
        info_out = llf.rvs(loc=mu_sample, scale=sd_llf)
        if not ppd:
            self.diary_out += [[info_out, t_sys]]
        return info_out


class NodeConjMu:
    """
    Nodes assuming Normal llf with known variance, able of belief-holding, -sampling, and -updating behavior via Normal conjugate prior.
    """

    def __init__(
        self,
        node_id,
        params_node=dict(
            loc=0,  # Current (prior) mean
            scale=10,  # Current (prior) standard deviation
        ),
        sd_llf=1,  # (Assumed known) standard deviation of llf
        diary_in=[],
        diary_out=[],
    ):
        """
        Initialize a Node capable of updating and sampling of a parameterized world model (= beliefs & log-probabilities of each belief).
        """
        self.node_id = node_id
        self.params_node = params_node.copy()
        self.sd_llf = sd_llf
        self.diary_in = diary_in.copy()
        self.diary_out = diary_out.copy()

    def set_updated_belief(self, info_in, id_in, t_sys):
        """Closed-form update for Normal llf (known variance) and Normal conjugate prior."""
        self.diary_in += [[info_in, id_in, t_sys]]
        self.params_node["loc"] = (
            self.params_node["scale"] ** 2 * info_in
            + self.sd_llf**2 * self.params_node["loc"]
        ) / (self.sd_llf**2 + self.params_node["scale"] ** 2)
        self.params_node["scale"] = (
            1 / (1 / self.params_node["scale"] ** 2 + 1 / self.sd_llf**2)
        ) ** 0.5

    def get_belief_sample(self, llf, t_sys):
        """
        Sample beliefs proportional to relative plausabilities.

        Returns a list of format [sample, t_sys].
        """
        info_out = llf.rvs(
            loc=self.params_node["loc"],
            scale=(self.params_node["scale"] ** 2 + self.sd_llf**2) ** 0.5,
        )
        self.diary_out += [[info_out, t_sys]]
        return info_out


####################################################################################################
# DEPRECATED, just here for pickle import compatibility
####################################################################################################
class Node:
    """
    Nodes with grid-wise belief-holding, -sampling, and -updating behavior.

    Attributes:
    beliefs -- Numpy array of possible parameter values into which a Node may hold belief
    log_probs -- Numpy array current relative log-probabilities of each belief
    llh -- method to return Bayesian log-likelihood p(data|parameters); supposed to take as first argument x=<data>
    diary_in -- Array of past incoming information
    diary_out -- Array of past outgoing information
    """

    def __init__(
        self,
        node_id,
        beliefs,
        log_priors,
        llh=st.norm(loc=0, scale=5),
        diary_in=[],
        diary_out=[],
    ):
        """
        Initialize a Node capable of updating and sampling of a grid-stored world model (= beliefs & log-probabilities of each belief).
        """
        assert len(beliefs) == len(log_priors)
        self.node_id = node_id
        self.beliefs = np.copy(beliefs)
        self.log_probs = np.copy(log_priors)
        self.llh = llh
        self.diary_in = np.array(diary_in.copy())
        self.diary_out = np.array(diary_out.copy())


#
#    def set_updated_belief(self, incoming_info):
#        """Bayesian update of the Node's belief."""
#
#        self.diary_in = np.append(self.diary_in, incoming_info)
#        self.log_probs += self.llh.logpdf(x=self.beliefs - incoming_info)
#        self.log_probs -= np.max(self.log_probs)  # subtract max for numerical stability
#
#    def get_belief_sample(self, size=1):
#        """Sample a belief according to world model."""
#
#        probs = logpdf_to_pdf(self.log_probs)
#        sample = np.random.choice(self.beliefs, size=size, p=probs)
#        self.diary_out = np.append(self.diary_out, sample)
####################################################################################################


class NodeNormal:
    """
    Nodes with grid-wise belief-holding, -sampling, and -updating behavior.

    Attributes:
    node_id : int
        Supposedly unique identifier of the node
    log_probs : array
        Unnormalized log-probabilities, corresponding to external 'beliefs' array
    params_node : dict
        Likelihood function paramters of node's model
    diary_in : list
        Past incoming information & metadata, structured [ [info_in_0, id_in_0, t_sys_0], [info_in_1, id_in_1, t_sys_1], ... ]
    diary_out : list
        Shared own information & metadata, structured [ [info_out_0, t_sys_0], [info_out_1, t_sys_1], ... ]
    """

    def __init__(
        self,
        node_id,
        log_priors,
        params_node=dict(  # Parameters defining the likelihood function (llf), normal distribution by default
            loc=0,
            scale=1,
        ),
        diary_in=[],
        diary_out=[],
    ):
        """
        Initialize a Node capable of updating and sampling of a grid-stored world model (= beliefs & log-probabilities of each belief).
        """
        self.node_id = node_id
        self.log_probs = np.copy(log_priors)
        self.params_node = params_node.copy()
        self.diary_in = diary_in.copy()
        self.diary_out = diary_out.copy()

    def set_updated_belief(self, llf_nodes, beliefs, info_in, id_in, t_sys):
        """Bayesian update of the Node's belief, based on incoming info 'info_in' from node with id 'id_in' at system time 't_sys'."""
        self.diary_in += [[info_in, id_in, t_sys]]
        self.log_probs += llf_nodes.logpdf(**self.params_node, x=beliefs - info_in)
        self.log_probs -= np.max(self.log_probs)  # subtract max for numerical stability

    def get_belief_sample(self, beliefs, t_sys):
        """
        Sample a belief "mu == <belief value>', proportional to relative plausabilities 'probs'.
        """
        probs = logpdf_to_pdf(self.log_probs)
        info_out = rng.choice(beliefs, p=probs)
        self.diary_out += [[info_out, t_sys]]
        return info_out
