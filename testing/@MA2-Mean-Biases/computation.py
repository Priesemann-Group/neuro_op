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


def model_run(in_tmp, name=""):
    """
    Call model with 'input0'.

    Serially run multiple model parameter sets, safe output to hdf5 file, garbage collect memory.
    """
    input = copy.deepcopy(in_tmp)
    print("Current run:\t", name)
    output = dict(nop.run_ConjMu(**input))
    print("\n\t t_exec = ", output["t_exec"], "s\n")
    with open("./input/in" + name + ".pkl", "wb") as f:
        pickle.dump(input, f)
    nop.export_hdf5(output, "./output/out" + name + ".h5")
    del output
    gc.collect()
    return None


input_ref = copy.deepcopy(nop.input_ref_ConjMu)
N_arr = [1, 2, 150]
sd_arr = np.round(np.arange(1 / 3, 2.1, 1 / 3), 2)
# input_ref["init_rngs"] = True
# input_ref["seed"] = 251328883828642274994245237017599543369

sd = sd_arr[idx]  # => 6 cores
for G in [
    nx.empty_graph(1),
    nop.build_random_network(N_nodes=2, N_neighbours=1),
    nop.build_random_network(N_nodes=150, N_neighbours=5),
]:
    in_tmp = copy.deepcopy(input_ref)
    if len(G) == 1:
        in_tmp["r"] = 0
    in_tmp["G"] = G
    in_tmp["node_params"]["scale"] = sd
    name = str("-N" + str(len(G)) + "-sd" + str(sd))
    model_run(in_tmp, name)
