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
- **with rng reinit of same seed** (after new rng init implementation)

### WallStreet1

- Coarse-grained scan ranges (especially for $\mu_{prior}$)
- **no rng reinit**

### WallStreet2

- Coarse-grained scan ranges (especially for $\mu_{prior}$)
- **with rng reinit of same seed** (after new rng init implementation)

### @MA-n1_amor

- Coarse scan of 3 orders of magnitudes for sd values
- with same seeds et al.
- aim: Plot of n1 behaviour for some priors for some oom.s (3d grid kld...) for MA