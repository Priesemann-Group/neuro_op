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
nn_arr = np.round(np.arange(4, 21, 4), 0)
sd_llf_arr = np.round(np.arange(0.5, 5.1, 0.5), 1)
input_ref["init_rngs"] = True
input_ref["seed"] = 251328883828642274994245237017599543369

sd_llf = sd_llf_arr[idx]  # => 10 cores
for N_nodes in N_arr:
    in_tmp = copy.deepcopy(input_ref)
    in_tmp["sd_llf"] = sd_llf
    if N_nodes == 1:
        in_tmp["G"] = nx.empty_graph(150)
        in_tmp["r"] = 0
        in_tmp["t_max"] = 1e4
        nn = 0
        name = str("-N" + str(N_nodes) + "-nn" + str(nn) + "-sd_llf" + str(sd_llf))
        model_run(in_tmp, name)
    elif N_nodes == 2:
        G = nx.empty_graph(150)
        for i in np.arange(0, 150, 2):
            G.add_edge(i, i + 1)
            in_tmp["G"] = G
        nn = 1
        name = str("-N" + str(N_nodes) + "-nn" + str(nn) + "-sd_llf" + str(sd_llf))
        model_run(in_tmp, name)
    # Only vary nn if N_Nodes is not 1,2
    else:
        for nn in nn_arr:
            in_tmp["G"] = nop.build_random_network(N_nodes=N_nodes, N_neighbours=nn)
            name = str("-N" + str(N_nodes) + "-nn" + str(nn) + "-sd_llf" + str(sd_llf))
            # model_run(in_tmp, name) # Well, I have this data already...
