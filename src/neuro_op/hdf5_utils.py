# HDF5 import/export functions for neuro_op package
import h5py
import networkx as nx
import numpy as np

from .classes import NodeGrid, NodeGridMu, NodeConjMu, NodeNormal


def export_hdf5(output, filename):
    """
    Export returns of 'run_model_Normal' to HDF5 file.

    Keyword arguments:
    output : dict
        Dictionary containing simulation results
    filename : str
        Name of to-be-created HDF5 file
    """

    with h5py.File(filename, "w") as f:
        # Store nodes, world
        for n in [
            "nodesGrid",
            "nodesGridMu",
            "nodesConjMu",
            "nodesNormal",
        ]:
            if n in output:
                nodes = f.create_group(n)
                for node in output[n] + [output["world"]]:
                    node_group = nodes.create_group("node_" + str(node.node_id))
                    node_group.create_dataset("node_id", data=node.node_id)
                    node_group.create_dataset("diary_in", data=np.array(node.diary_in))
                    node_group.create_dataset(
                        "diary_out", data=np.array(node.diary_out)
                    )
                    if n == "nodesGrid":
                        node_group.create_dataset("log_probs", data=node.log_probs)
                    if n == "nodesGridMu":
                        node_group.create_dataset("log_probs", data=node.log_probs)
                    if n == "nodesConjMu":
                        node_group.create_dataset("sd_llf", data=node.sd_llf)
                        for key, value in node.params_node.items():
                            node_group.create_dataset(f"params_node/{key}", data=value)
                    if n == "nodesNormal":
                        node_group.create_dataset("log_probs", data=node.log_probs)
                        for key, value in node.params_node.items():
                            node_group.create_dataset(f"params_node/{key}", data=value)

        # Store rest of output, see key strings...
        f.create_dataset("G", data=nx.to_numpy_array(output["G"]))
        f.create_dataset("N_events", data=output["N_events"])
        f.create_dataset("t_end", data=output["t_end"])
        f.create_dataset("t_start", data=output["t_start"])
        f.create_dataset("t_exec", data=output["t_exec"])
        f.create_dataset("seed", data=str(output["seed"]))
        if "nodesGrid" in output:
            f.create_dataset("mu_arr", data=output["mu_arr"])
            f.create_dataset("sd_arr", data=output["sd_arr"])
        if "nodesGridMu" in output:
            f.create_dataset("mu_arr", data=output["mu_arr"])
        if "nodesConjMu" in output:
            pass  # Nothing else to store
        if "nodesNormal" in output:
            f.create_dataset("beliefs", data=output["beliefs"])
        if "mu_nodes" in output:
            f.create_dataset("mu_nodes", data=np.array(output["mu_nodes"]))
        if "kl_divs" in output:
            f.create_dataset("kl_divs", data=np.array(output["kl_divs"]))


def import_hdf5(filename):
    """
    Import simulation results from HDF5 file.

    Keyword arguments:
    filename : str
        Name of HDF5 file to be imported

    Returns:
    Dictionary containing simulation results
    """

    with h5py.File(filename, "r") as f:
        # Create nodes, world objects from imported old data...
        nodes = []
        world = None
        if "nodesGrid" in f:
            n = "nodesGrid"
            for node_name in f[n]:
                node_group = f[n][node_name]
                if node_group["node_id"][()] < 0:
                    world = NodeGrid(
                        node_group["node_id"][()],
                        node_group["log_probs"][()],
                        node_group["diary_in"][()],
                        node_group["diary_out"][()],
                    )
                else:
                    nodes.append(
                        NodeGrid(
                            node_group["node_id"][()],
                            node_group["log_probs"][()],
                            node_group["diary_in"][()],
                            node_group["diary_out"][()],
                        )
                    )
        elif "nodesGridMu" in f:
            n = "nodesGridMu"
            for node_name in f[n]:
                node_group = f[n][node_name]
                if node_group["node_id"][()] < 0:
                    world = NodeGridMu(
                        node_group["node_id"][()],
                        node_group["log_probs"][()],
                        node_group["diary_in"][()],
                        node_group["diary_out"][()],
                    )
                else:
                    nodes.append(
                        NodeGridMu(
                            node_group["node_id"][()],
                            node_group["log_probs"][()],
                            node_group["diary_in"][()],
                            node_group["diary_out"][()],
                        )
                    )
        elif "nodesConjMu" in f:
            n = "nodesConjMu"
            for node_name in f[n]:
                node_group = f[n][node_name]
                if node_group["node_id"][()] < 0:
                    world = NodeConjMu(
                        node_group["node_id"][()],
                        {
                            key: node_group[f"params_node/{key}"][()]
                            for key in node_group["params_node"]
                        },
                        node_group["sd_llf"][()],
                        node_group["diary_in"][()],
                        node_group["diary_out"][()],
                    )
                else:
                    nodes.append(
                        NodeConjMu(
                            node_group["node_id"][()],
                            {
                                key: node_group[f"params_node/{key}"][()]
                                for key in node_group["params_node"]
                            },
                            node_group["sd_llf"][()],
                            node_group["diary_in"][()],
                            node_group["diary_out"][()],
                        )
                    )
        elif "nodesNormal" in f:
            n = "nodesNormal"
            for node_name in f[n]:
                node_group = f[n][node_name]
                if node_group["node_id"][()] < 0:
                    world = NodeNormal(
                        node_group["node_id"][()],
                        node_group["log_probs"][()],
                        {
                            key: node_group[f"params_node/{key}"][()]
                            for key in node_group["params_node"]
                        },
                        node_group["diary_in"][()],
                        node_group["diary_out"][()],
                    )
                else:
                    nodes.append(
                        NodeNormal(
                            node_group["node_id"][()],
                            node_group["log_probs"][()],
                            {
                                key: node_group[f"params_node/{key}"][()]
                                for key in node_group["params_node"]
                            },
                            node_group["diary_in"][()],
                            node_group["diary_out"][()],
                        )
                    )
        else:
            print("No known node class referenced")

        # Import rest of exported output; add nodes to output dictionary...
        dict_out = dict(
            world=world,
            G=nx.from_numpy_array(f["G"][()]),
            N_events=f["N_events"][()],
            t_end=f["t_end"][()],
            t_start=f["t_start"][()],
            t_exec=f["t_exec"][()],
            seed=f["seed"][()],
        )
        if "nodesGrid" in f:
            dict_out["nodesGrid"] = nodes
            dict_out["mu_arr"] = f["mu_arr"][()]
            dict_out["sd_arr"] = f["sd_arr"][()]
        elif "nodesGridMu" in f:
            dict_out["nodesGridMu"] = nodes
            dict_out["mu_arr"] = f["mu_arr"][()]
        elif "nodesConjMu" in f:
            dict_out["nodesConjMu"] = nodes
        elif "nodesNormal" in f:
            dict_out["nodesNormal"] = nodes
            dict_out["beliefs"] = f["beliefs"][()]
        else:
            print(
                "As no nodes were given, the returned dictionary has no nodes as well."
            )
        if "mu_nodes" in f:
            dict_out["mu_nodes"] = f["mu_nodes"][()]
        if "kl_divs" in f:
            dict_out["kl_divs"] = f["kl_divs"][()]

        return dict_out
