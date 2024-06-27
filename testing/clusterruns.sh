import copy  # deep-copying of input dictionary (which includes mutable objects)
import gc  # explicit garbace collection calling after each run
import itertools  # for iterating over multiple parameter sets
import pickle  # saving input dictionary to file
import neuro_op as nop  # project's main module
import networkx as nx  # graph library
import numpy as np
import scipy.stats as st
import sys

if len(sys.argv) > 1:
    idx = int(sys.argv[1]) - 1

input_ref = dict(
    G=nx.empty_graph(1).to_directed(),
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
    t_sample=1,  # Periodicity for which samples and distance measures (KL-div, p-distance) are taken
    sample_bins=401,  # Number of bins used in distance measures
    sample_range=(
        -20,
        20,
    ),  # Interval over which distance measure distributions are considered
    sampling=True,  # Switch for sampling
)


def model_run(input0, name=""):
    """
    Call model with 'input0'.

    Serially run multiple model parameter sets, safe output to hdf5 file, garbage collect memory.
    """

    input = copy.deepcopy(input0)
    print("Current run:\t", name)
    output = dict(nop.run_model_Param(**input))
    print("\n\t t_exec = ", output["t_exec"], "s\n")
    with open("in" + name + ".pkl", "wb") as f:
        pickle.dump(input, f)
    nop.export_hdf5(output, "out" + name + ".h5")
    del output
    gc.collect()
    return None


# input0 = copy.deepcopy(input_ref)
# mu_arr = np.arange(0, 5.1, 1)
# sd_arr = [0.5, 1, 2, 5, 10]
# for mu, sd in itertools.product(mu_arr, sd_arr):
#    input0["G"] = nx.empty_graph(1).to_directed()
#    input0["t_max"] = 10000
#    input0["r"] = 0
#    input0["params_node"]["loc"] = mu
#    input0["params_node"]["scale"] = sd
#    name = str("N1-mu-" + str(mu) + "-sd-" + str(sd))
#    model_run(input0, name)
#
# input0 = copy.deepcopy(input_ref)
# nn_arr = np.arange(5, 100, 5)
# mu_arr = np.arange(0, 5.1, 1)
# sd_arr = [0.5, 1, 2, 5, 10]
# r_arr = np.arange(0.25, 5.1, 0.25)
# for nn, mu, sd, r in itertools.product(nn_arr, mu_arr, sd_arr, r_arr):
#    input0["G"] = nop.build_random_network(100, nn)
#    name = str("nn-" + str(nn) + "mu-" + str(mu) + "-sd-" + str(sd) + "-r-" + str(r))
#    model_run(input0, name)
#
input0 = copy.deepcopy(input_ref)
nn_arr = np.arange(5, 100, 5)
mu_arr = np.arange(0, 5.1, 1)
sd_arr = [0.5, 1, 2, 5, 10]
r_arr = np.arange(0.25, 5.1, 0.25)
nn, mu, sd = list(itertools.product(nn_arr, mu_arr, sd_arr))[idx]

input0["G"] = nop.build_random_network(100, nn)
input0["params_node"]["loc"] = mu
input0["params_node"]["scale"] = sd
for r in r_arr:
    input0["r"] = r
    name = str("-nn" + str(nn) + "-mu" + str(mu) + "-sd" + str(sd) + "-r" + str(r))
    model_run(input0, name)
