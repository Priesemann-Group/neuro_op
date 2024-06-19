import copy  # deep-copying of input dictionary (which includes mutable objects)
import gc  # explicit garbace collection calling after each run
import neuro_op as nop  # project's main module
import numpy as np
import pickle  # output export/import
import scipy.stats as st
import time  # runtime measuring


def model_runs(input0, dict_list, string_list):
    """
    Call 'run_model' with 'input_standard' adapted at specified dictionary entries.

    Serially run multiple model parameter sets, safe output to pickle file, garbage collect memory.
    """

    assert len(dict_list) == len(string_list)

    for i, _ in enumerate(dict_list):
        input = copy.deepcopy(input0)
        for key, value in dict_list[i].items():
            input[key] = value
        print("Current adaptions:\t", string_list[i])
        t0 = time.time()
        output = dict(nop.run_model(**input))
        t_exec = time.time() - t0
        output["t_start"] = time.strftime("%Y-%m-%d--%H-%M", time.localtime(t0))
        output["t_exec"] = t_exec
        print("t_exec = ", t_exec, "s")
        with open("out--" + string_list[i] + ".pkl", "wb") as f:
            pickle.dump(output, f)
        del output
        gc.collect()


input0 = copy.deepcopy(nop.input_standard)
input0["sample_bins"] = 101
input0["sample_opinion_range"] = (-50, 50)

variations = []
var_string = []
# Power laws: mean if (a-1)<-2, variance if (a-1)<-3
# (or: a<-1, a<-2) -- BUT st.powerlaw wants a>0
for a_tmp in [10, 50, 85]:

    def world_pow(x):
        return st.powerlaw(a=a_tmp / 100, scale=50).logpdf(x=np.abs(x))

    for N in [1.5, 2.5]:
        for r in [0.1, 1, 10]:
            for op_min in [0, -50]:
                variations.append(
                    dict(
                        N_nodes=int(10**N),
                        belief_min=op_min,
                        world_logpdf=world_pow,
                        r=r,
                        sample_opinion_range=(
                            op_min,
                            input0["sample_opinion_range"][1],
                        ),
                    )
                )
                var_string.append(
                    world_pow.__name__
                    + "--a-"
                    + str(a_tmp)
                    + "--op_min-"
                    + str(op_min)
                    + "--N-"
                    + str(N)
                    + "--r-"
                    + str(r)
                )


for N in [1.5, 2.5, 3.5]:
    for r in [0.1, 1, 10]:
        for prior in [0, 1]:
            if prior:
                small_prior = nop.dist_binning(
                    logpdf=st.norm(loc=0, scale=5).logpdf,
                    N_bins=len(input0["log_priors"]),
                    range=input0["sample_opinion_range"],
                )
                variations.append(dict(N_nodes=int(10**N), log_priors=small_prior, r=r))
                var_string.append("N-" + str(N) + "--r-" + str(r) + "--small-prior")
            else:
                variations.append(dict(N_nodes=int(10**N), r=r))
                var_string.append("N-" + str(N) + "--r-" + str(r) + "--uniform-prior")


model_runs(input0, variations, var_string)

with open("input0.pkl", "wb") as f:
    pickle.dump(input0, f)
