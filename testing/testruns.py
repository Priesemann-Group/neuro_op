import copy  # deep-copying of input dictionary (which includes mutable objects)
import gc  # explicit garbace collection calling after each run
import neuro_op as nop  # project's main module
import numpy as np
import pickle  # output export/import
import scipy.stats as st
import time  # runtime measuring


def model_runs(input0, dict_list, numero=""):
    """
    Call simulate model with 'input0'.

    Serially run multiple model parameter sets, safe output to hdf5 file, garbage collect memory.
    """

    for dic_tmp in dict_list:
        input = copy.deepcopy(input0)
        adaptions = ""
        for key, value in dic_tmp.items():
            input[key] = value
            adaptions += "--" + str(key) + "-" + str(value)
        print("Current adaptions:\t", dic_tmp.items())
        t0 = time.time()
        output = dict(nop.run_model_Grid(**input))
        t1 = time.time()
        print("For adaptions\t", dic_tmp.items(), " :\n\t t_exec = ", (t1 - t0))
        with open("in" + numero + adaptions + ".pkl", "wb") as f:
            pickle.dump(input, f)
        nop.import_hdf5_Grid(output, "out" + numero + adaptions + ".h5")
        del output
        gc.collect()


input0 = dict(
    G=nop.build_random_network(N_nodes=100, N_neighbours=5),  # networkx graph object
    llf_nodes=st.norm,  # Likelihood function (llf) of nodes, Gaussian by default
    llf_world=st.norm,  # Likelihood function (llf) of to-be-approximated world state, Gaussian by default
    params_node=dict(  # Likelihood function (llf) parameters of nodes, Gaussian by default
        loc=0,
        scale=1,
    ),
    params_world=dict(  # Likelihood function (llf) parameters of to-be-approximated world state, Gaussian by default
        loc=0,
        scale=1,
    ),
    beliefs=np.linspace(  # beliefs considered by each node
        start=-10,  # min. considered belief value
        stop=10,  # max. considered belief value
        num=500,  # number of considered belief values
    ),
    log_priors=np.zeros(500),  # Prior log-probabilities of nodes
    # Dynamics parameters (rates, simulation times)...
    h=1,
    r=1,
    t0=0,
    t_max=20,
    # Sampling parameters...
    t_sample=0.5,
    sample_bins=101,
    sample_range=(-10, 10),
    p_distance_params=[(1, 1), (2, 1)],
    # Switches...
    progress=False,
)

for numero in range(5):
    variations = []
    for N in [1.5, 2.5]:
        G = nop.build_random_network(N_nodes=int(10**N), N_neighbours=5)
        input0["G"] = G
        for r_by_h in list(np.linspace(0.0, 1.0, 11)) + list(np.linspace(1.5, 5.0, 9)):
            variations.append(
                dict(N_nodes=int(10**N), r=r_by_h)
            )  # r = r_by_h * h, but h=1 by default
    model_runs(input0, variations, numero)
