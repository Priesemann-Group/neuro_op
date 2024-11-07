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
input_ref["t_max"] = 1e3
N_arr = [1, 2, 150]
sd_arr = np.round(np.arange(1 / 3, 2.1, 1 / 3), 2)
G = nop.build_random_network(
    150, 10
)  # Just called N_neighbours=10 sensible, nothing else to it
# input_ref["init_rngs"] = True
# input_ref["seed"] = 251328883828642274994245237017599543369

sd = sd_arr[idx]  # => 6 cores
for N in N_arr:
    in_tmp = copy.deepcopy(input_ref)
    if N == 1:
        in_tmp["G"] = nx.empty_graph(150)
        in_tmp["r"] = 0
    elif N == 2:
        G2 = nx.empty_graph(150)
        [G2.add_edge(i, i + 1) for i in np.arange(0, 150, 2)]
        in_tmp["G"] = G2
    else:
        in_tmp["G"] = G
    in_tmp["params_node"]["scale"] = sd
    name = str("-N" + str(N) + "-sd" + str(sd))
    model_run(in_tmp, name)
