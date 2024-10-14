# Log

Rough description of directories and their contents.
Unless noted otherwise, each test used `NodesConjMu`.
Squared brackets show parameters scanned for.
Normal brackets show dependencies to plot.

---


## MA1-FirstBuilder

$\approx$ Fig. 1 of *Baumann et al.*
- aka. $\mu_i[\sigma_{in}, \beta](t)$
    - $\beta := 1-N_{neighb}/N \approx$ homophily
    - $\sigma_{in} \approx$ (lack of) trust into incoming data // controversialness // convergence pressure

## MA2-Mean-Biases

$\approx$ Fig. 2 of *Baumann et al.*
- aka. $\langle KLD_i // |\mu_i|\rangle _i[\sigma_{in}, \sigma_{prior}] (t_{max})$
- $\sigma_{in} \approx$ convergence pressure (as above)
- $\sigma_{prior} \approx$ worth of news // *interaction strength* (fishy and very similar to $\sigma_in$ in interpretation... this too shall pass)
- $t_{max}$: some arbitrary simulation end (or rather sample point during run)

## MA3-Mean-Biases

$\approx$ Fig. 3/4 of *Baumann et al*
- aka. density plots $\rho(\bullet)$ of opinions/KLD:
    - $\rho(r, \mu_i)$
        - $r \approx$ 
    - $\rho(\langle \mu_j \rangle _{j: \exists w_{ij}}, \mu_i)$


---


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

## @MA-n1_amor

- Coarse scan of 3 orders of magnitudes for sd values
- with same seeds et al.
- aim: Plot of n1 behaviour for some priors for some oom.s (3d grid kld...) for MA