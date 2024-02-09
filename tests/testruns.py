import copy  # deep-copying of input dictionary (which includes mutable objects)
import gc  # explicit garbace collection calling after each run
import neuro_op as nop  # project's main module
import pickle  # output export/import
import time  # runtime measuring


def model_scan(dict_list):
    """
    Call 'run_model' with 'input_standard' adapted at specified dictionary entries.

    Serially run multiple model parameter sets, safe output to pickle file, garbage collect memory.
    """

    for dic_tmp in dict_list:
        input = copy.deepcopy(nop.input_standard)
        for key, value in dic_tmp.items():
            input[key] = value
        print("Current adaptions:\t", dic_tmp.items())
        t0 = time.time()
        output = dict(nop.run_model(**input))
        t1 = time.time()
        output["t_exec"] = t1 - t0
        print("For adaptions\t", dic_tmp.items(), " :\n\t t_exec = ", (t1 - t0))
        filename = (
            "out"
            + str(dic_tmp)
            + time.strftime("--%Y-%m-%d--%H-%M--", time.localtime(t0))
            + "--export.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(output, f)
        del output
        gc.collect()


params = []

for N in [1, 2, 3, 4]:
    for t in [1, 2, 3, 4]:
        params.append(dict(N_nodes=10**N, t_max=10**t))

for t in [5, 6]:
    params.append(dict(t_max=10**t))

for r, h in [(0.1, 5), (5, 0.1), (5, 5)]:
    for N in [1, 2, 3, 4]:
        params.append(dict(N_nodes=10**N, r=r, h=h))

for N in [5]:
    params.append(dict(N_nodes=10**N))


model_scan(params)
