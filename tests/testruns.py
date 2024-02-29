import copy  # deep-copying of input dictionary (which includes mutable objects)
import gc  # explicit garbace collection calling after each run
import neuro_op as nop  # project's main module
import numpy as np
import pickle  # output export/import
import time  # runtime measuring


def model_scan(dict_list):
    """
    Call 'run_model' with 'input_standard' adapted at specified dictionary entries.

    Serially run multiple model parameter sets, safe output to pickle file, garbage collect memory.
    """

    for dic_tmp in dict_list:
        input = copy.deepcopy(nop.input_standard)
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


params = []

for N in np.arange(1.0, 4.5, 0.5):
    for t in np.arange(1.0, 5.0, 0.5):
        params.append(dict(N_nodes=int(10**N), t_max=int(10**t)))

for t in np.arange(5.0, 6.5, 0.5):
    params.append(dict(t_max=int(10**t)))

for r, h in [(0.1, 5), (5, 0.1), (5, 5)]:
    for N in np.arange(1.0, 4.5, 0.5):
        params.append(dict(N_nodes=int(10**N), r=r, h=h))

for N in np.arange(4.5, 5.5, 0.5):
    params.append(dict(N_nodes=int(10**N)))


model_scan(params)
