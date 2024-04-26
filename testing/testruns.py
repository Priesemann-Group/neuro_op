import copy  # deep-copying of input dictionary (which includes mutable objects)
import gc  # explicit garbace collection calling after each run
import neuro_op as nop  # project's main module
import numpy as np
import pickle  # output export/import
import scipy.stats as st
import time  # runtime measuring


def model_run(input0, name=""):
    """
    Call simulate model with 'input0'.

    Serially run multiple model parameter sets, safe output to hdf5 file, garbage collect memory.
    """

    input = copy.deepcopy(input0)
    print("Current adaptions:\t", name)
    start = time.time()
    output = dict(nop.run_model_Grid(**input))
    t1 = time.time()
    print("\n\t t_exec = ", output["t_exec"], "s\n")
    with open("in" + name + ".pkl", "wb") as f:
        pickle.dump(input, f)
    nop.export_hdf5_Grid(output, "out" + name + ".h5")
    del output
    gc.collect()


input_ref = dict(
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
    t_max=50,
    # Sampling parameters...
    t_sample=0.5,
    sample_bins=101,
    sample_range=(-10, 10),
    p_distance_params=[(1, 1), (2, 1)],
    # Switches...
    progress=False,
)

input0 = copy.deepcopy(input_ref)
for numero in range(5):
    for N in [1.5, 2.5]:
        input0["G"] = nop.build_random_network(N_nodes=int(10**N), N_neighbours=5)
        for r_by_h in list(np.linspace(0.0, 1.0, 11)) + list(np.linspace(1.5, 5.0, 9)):
            input0["r"] = r_by_h
            name = str(numero) + "--N-" + str(N) + "--r-" + str(round(r_by_h, 1))
            model_run(input0, name)
