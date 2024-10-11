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

input_ref = nop.input_ref_ConjMu
input_ref["G"] = nx.empty_graph(1).to_directed()
input_ref["r"] = 0
input_ref["t_max"] = 100000
input_ref["t_sample"] = 1
input_ref["sample_range"] = (-20, 20)
input_ref["sample_bins"] = 801
input_ref["init_rngs"] = True
input_ref["seed"] = 251328883828642274994245237017599543369


def model_run(input0, name=""):
    """
    Call model with 'input0'.

    Serially run multiple model parameter sets, safe output to hdf5 file, garbage collect memory.
    """

    input = copy.deepcopy(input0)
    print("Current run:\t", name)
    output = dict(nop.run_ConjMu(**input))
    print("\n\t t_exec = ", output["t_exec"], "s\n")
    with open("in" + name + ".pkl", "wb") as f:
        pickle.dump(input, f)
    nop.export_hdf5(output, "out" + name + ".h5")
    del output
    gc.collect()
    return None


mu_arr = np.round(np.arange(0, 11, 1), 0)
sd_arr = [0.2, 1, 2]
mu = mu_arr[idx]
# mu, sd = list(itertools.product(mu_arr, sd_arr))[idx]

for sd in sd_arr:
    for sd2 in sd_arr:
        input0 = copy.deepcopy(input_ref)
        input0["params_node"]["loc"] = mu
        input0["params_node"]["scale"] = sd
        input0["params_world"]["scale"] = sd2
        name = str("-sdw" + str(sd2) + "-mu" + str(mu) + "-sd" + str(sd))
        model_run(input0, name)

        input0 = copy.deepcopy(input_ref)
        input0["params_node"]["loc"] = mu
        input0["params_node"]["scale"] = sd
        input0["sd_llf"] = sd2
        name = str("-sd_llf" + str(sd2) + "-mu" + str(mu) + "-sd" + str(sd))
        model_run(input0, name)
