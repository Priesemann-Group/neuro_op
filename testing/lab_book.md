# Log

Rough description of directories and their contents.
Unless noted otherwise, each test used `NodesConjMu`.

## @n1_amor

- Parameter scans for one node, using NodesParam

### BlackJack1

- Fine-grained scan ranges (usually step size 0.2)
- **no rng reinit**

### BlackJack2

- Fine-grained scan ranges (usually step size 0.2)
- **with rng reinit** (after new rng init implementation)

### WallStreet1

- Coarse-grained scan ranges (especially for $\mu_{prior}$)
- **no rng reinit**

### WallStreet2

- Coarse-grained scan ranges (especially for $\mu_{prior}$)
- **with rng reinit** (after new rng init implementation)
