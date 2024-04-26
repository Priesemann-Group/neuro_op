import copy  # deep-copying of input dictionary (which includes mutable objects)
import gc  # explicit garbace collection calling after each run
import neuro_op as nop  # project's main module
import numpy as np
import pickle  # output export/import
import scipy.stats as st
import time  # runtime measuring


def model_runs(input0, dict_list):
    """
    Call 'run_model' with 'input_standard' adapted at specified dictionary entries.

    Serially run multiple model parameter sets, safe output to pickle file, garbage collect memory.
    """

    for dic_tmp in dict_list:
        input = copy.deepcopy(input0)
        adaptions = ""
        for key, value in dic_tmp.items():
            input[key] = value
            adaptions += "--" + str(key) + "-" + str(value)
        print("Current adaptions:\t", dic_tmp.items())
        t0 = time.time()
        output = dict(nop.run_model(**input))
        t1 = time.time()
        output["t_start"] = time.strftime("%Y-%m-%d--%H-%M", time.localtime(t0))
        output["t_exec"] = t1 - t0
        print("For adaptions\t", dic_tmp.items(), " :\n\t t_exec = ", (t1 - t0))
        filename = "out" + adaptions + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(output, f)
        del output
        gc.collect()


input0 = copy.deepcopy(nop.input_standard)
input0["t_max"] = 1000
input0["sample_bins"] = 101

variations = []
# Power laws: mean if (a-1)<-2, variance if (a-1)<-3
# (or: a<-1, a<-2) -- BUT st.powerlaw wants a>0
for a_tmp in [10, 50, 85]:

    def world_pow(x):
        return st.powerlaw(a=a_tmp / 100, scale=50).logpdf(x=np.abs(x))

    for N in [2, 3]:
        for r, h in [(1, 1), (0.1, 5), (5, 0.1)]:
            variations.append(
                dict(N_nodes=int(10**N), h=h, r=r, world_logpdf=world_pow)
            )

model_runs(input0, variations)
