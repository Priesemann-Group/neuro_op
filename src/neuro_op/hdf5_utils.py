# HDF5 import/export functions for neuro_op package
import h5py
import networkx as nx
import numpy as np

from .classes import NodeNormal


def export_hdf5_Grid(output, filename):
    """
    Export returns of 'run_model_Normal' to HDF5 file.

    Keyword arguments:
    output : dict
        Dictionary containing simulation results:
            nodes       <class 'list'>
            G           <class 'networkx.classes.digraph.DiGraph'>
            beliefs     <class 'numpy.ndarray'>
            world       <class 'neuro_op.neuro_op.NodeNormal'>
            N_events    <class 'int'>
            t_end       <class 'numpy.float64'>
            mu_nodes    <class 'list'>
            kl_divs     <class 'list'>
            p_distances <class 'list'>
            seed        <class 'int'>
    filename : str
        Name of to-be-created HDF5 file
    """

    with h5py.File(filename, "w") as f:
        # nodes, world
        for n in [
            "nodesGrid",
            "nodesNormal",
            "nodesConj",
        ] and n in output:
            #### HERE WE ARE ... #################################
            nodes = f.create_group(n)
            for node in nodelist:
                node_group = nodes.create_group("node_" + str(node.node_id))
                node_group.create_dataset("node_id", data=node.node_id)
                if n in ["nodesGrid, nodesNormal"]:
                    node_group.create_dataset("log_probs", data=node.log_probs)
                if n in ["nodesNormal", "nodesConj"]:
                    for key, value in node.params_node.items():
                        node_group.create_dataset(f"params_node/{key}", data=value)
                if n in ["nodesConj"]:
                    node_group.create_dataset("sd_llf", data=node.sd_llf)
                node_group.create_dataset("diary_in", data=np.array(node.diary_in))
                node_group.create_dataset("diary_out", data=np.array(node.diary_out))

        # rest of it, see key strings...
        f.create_dataset("G", data=nx.to_numpy_array(output["G"]))
        f.create_dataset("beliefs", data=output["beliefs"])
        f.create_dataset("N_events", data=output["N_events"])
        f.create_dataset("t_end", data=output["t_end"])
        f.create_dataset("mu_nodes", data=np.array(output["mu_nodes"]))
        f.create_dataset("kl_divs", data=np.array(output["kl_divs"]))
        f.create_dataset("p_distances", data=np.array(output["p_distances"]))
        f.create_dataset("t_start", data=output["t_start"])
        f.create_dataset("t_exec", data=output["t_exec"])
        f.create_dataset("seed", data=str(output["seed"]))


def import_hdf5_Grid(filename):
    """
    Import simulation results from HDF5 file.

    Keyword arguments:
    filename : str
        Name of HDF5 file to be imported

    Returns:
    Dictionary containing simulation results:
        nodes       <class 'list'>
        G           <class 'networkx.classes.digraph.DiGraph'>
        beliefs     <class 'numpy.ndarray'>
        world       <class 'neuro_op.neuro_op.NodeNormal'>
        N_events    <class 'int'>
        t_end       <class 'numpy.float64'>
        mu_nodes    <class 'list'>
        kl_divs     <class 'list'>
        p_distances <class 'list'>
        seed        <class 'int'>
    """

    with h5py.File(filename, "r") as f:

        # nodes, world
        nodes = []
        world = None
        for node_name in f["nodes"]:
            node_group = f["nodes"][node_name]
            if node_group["node_id"][()] == -1:
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

        # Rest of it, see key strings...
        G = nx.from_numpy_array(f["G_dict"][()])
        beliefs = f["beliefs"][()]
        N_events = f["N_events"][()]
        t_end = f["t_end"][()]
        mu_nodes = f["mu_nodes"][()]
        kl_divs = f["kl_divs"][()]
        p_distances = f["p_distances"][()]
        t_start = f["t_start"][()]
        t_exec = f["t_exec"][()]
        seed = f["seed"][()]

    return {
        "nodes": nodes,
        "G": G,
        "beliefs": beliefs,
        "world": world,
        "N_events": N_events,
        "t_end": t_end,
        "mu_nodes": mu_nodes,
        "kl_divs": kl_divs,
        "p_distances": p_distances,
        "t_start": t_start,
        "t_exec": t_exec,
        "seed": seed,
    }
