import copy  # deep-copying of input dictionary (which includes mutable objects)
import gc  # explicit garbace collection calling after each run
import pickle  # saving input dictionary to file
import neuro_op as nop  # project's main module
import numpy as np
import scipy.stats as st


input_ref = dict(
    G=nop.build_random_network(N_nodes=100, N_neighbours=5),  # networkx graph object
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
    t_sample=0.5,  # Periodicity for which samples and distance measures (KL-div, p-distance) are taken
    sample_bins=201,  # Number of bins used in distance measures
    sample_range=(
        -5,
        5,
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


input0 = copy.deepcopy(input_ref)
for sd in [10, 0.5, 1.0, 1.5, 2.0]:
    input0["params_node"]["scale"] = sd
    for mu in np.arange(0.0, 2.51, 0.5):
        input0["params_node"]["loc"] = mu
    for r_by_h in np.arange(0.0, 5.01, 0.25):
        input0["r"] = r_by_h
        name = str(
            "--mu-" + str(mu) + "--sd-" + str(sd) + "--r-" + str(round(r_by_h, 2))
        )
        model_run(input0, name)
