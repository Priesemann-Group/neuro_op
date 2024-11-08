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
N_nodes = [1, 2, 150]
mu_arr = np.round(np.arange(0, 10.1, 2.5), 0)
sd_arr = np.round(np.arange(1, 10.1, 2), 1)
r_arr = np.round(np.arange(1, 10.1, 2), 1)
input_ref["init_rngs"] = True
input_ref["seed"] = 169009300480314836251067998491130068212
G = nop.build_random_network(150, 5)
G2 = nx.empty_graph(150)
[G2.add_edge(i, i + 1) for i in np.arange(0, 150, 2)]

N_nodes, mu = list(itertools.product(N_nodes, mu_arr))[idx]  # => 15 cores
in_tmp = copy.deepcopy(input_ref)
in_tmp["params_node"]["loc"] = mu
if N_nodes == 1:
    in_tmp["G"] = nx.empty_graph(150)
    in_tmp["r"] = 0  # 1 node, so has to be
    for sd in sd_arr:
        in_tmp["params_node"]["scale"] = sd
        r = np.round(0, 1)
        name = str(
            "-N" + str(N_nodes) + "-r" + str(r) + "-mu" + str(mu) + "-sd" + str(sd)
        )
        model_run(in_tmp, name)
else:
    for sd, r in itertools.product(sd_arr, r_arr):
        in_tmp["params_node"]["scale"] = sd
        in_tmp["r"] = r
        if N_nodes == 2:
            in_tmp["G"] = G2
        else:
            in_tmp["G"] = nop.build_random_network(N_nodes=N_nodes, N_neighbours=5)

        name = str(
            "-N" + str(N_nodes) + "-r" + str(r) + "-mu" + str(mu) + "-sd" + str(sd)
        )
        model_run(in_tmp, name)
