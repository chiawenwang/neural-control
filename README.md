# Toy DER (branch `der`)

This branch holds a self-contained, from-scratch implementation of
**Discrete Elastic Rods** (Bergou et al., ACM TOG 2008), used to validate the basic
DER model: stretching + bending + twisting energies, a Newton static solver and a
backward-Euler dynamic step. It has no contact and no control.

The Neural Control code lives on `main`; nothing here depends on it.

```
toy_case/        # the DER model, solvers, experiments — see toy_case/README.md
reference/       # Discrete elastic rods.pdf
requirement.yml  # conda environment (torch, numpy, matplotlib, pyyaml)
```

## Quick start

```bash
conda env create -f requirement.yml   # or use any env with torch, numpy, matplotlib, pyyaml
conda activate der
cd toy_case
python run.py gradcheck    # force correctness + comparison with the paper's closed forms
python run.py all          # cantilever, twist, buckle, swing; figures go to toy_case/outputs/
```

Full documentation of the model, experiments and config: [`toy_case/README.md`](toy_case/README.md).
