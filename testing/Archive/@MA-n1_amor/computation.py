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
    with open("in" + name + ".pkl", "wb") as f:
        pickle.dump(input, f)
    nop.export_hdf5(output, "out" + name + ".h5")
    del output
    gc.collect()
    return None


input_ref = copy.deepcopy(nop.input_ref_ConjMu)
input_ref["t_max"] = 500
# input_ref["init_rngs"] = True
# input_ref["seed"] = 251328883828642274994245237017599543369


nn_arr = np.concatenate((np.arange(0, 10, 2), np.arange(10, 100, 10)))
for nn in nn_arr:
    in_tmp = copy.deepcopy(input_ref)
    in_tmp["G"] = nop.build_random_network(N_nodes=100, N_neighbours=nn)
    name = str("-NN" + str(nn))
    model_run(in_tmp, name)
