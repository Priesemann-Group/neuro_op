# neuro_op
Package to research information establishment in networks of topology-restricted active inference nodes.

## Installation 

### From source

Clone this repository via HTTPS

```bash
$ git clone https://github.com/Priesemann-Group/neuro_op.git
```

or SSH (requires SSH key setup)

```bash
$ git clone git@github.com:Priesemann-Group/neuro_op.git
```

Subsequently, change into the just-cloned directory and, if you have not yet set up a conda environment with needed packages, do so:

``bash
$ cd neuro_op
$ # If not yet existing, create conda environment with needed packages. This also installs pip within that environment
$ conda env create -f environment.yml
```

To install the package within this environment (named `neuro_op`), activate the package and install from the repo's directory

```bash
$ conda activate neuro_op
$ pip install -e .	    # This assumes you have been 'cd'ing into the neuro_op directory
```

This will install the project in editable mode, meaning that changes in the local source code will apply without requiring a reinstall of the package.

To access this environment via Jupyter (e.g., via JupyterLab), install an ipykernel for it:

```bash
$ python -m ipykernel install --user --name neuro_op --display-name "Python 3.11 (neuro_op)"
```
