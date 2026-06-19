# km-update-tomography

Code accompanying the paper **"Connecting Quantum Tomography and Quantum Retrodiction."**

This repository implements iterative Petz and Kubo–Mori (KM) updates for maximum-likelihood quantum state reconstruction from measurement data, including Pauli tomography and tomography through general quantum channels in a Stinespring framework.

## Files

### `petz_pauli_tomography.py`

Maximum-likelihood quantum state tomography based on iterative weighted Petz updates using overcomplete Pauli measurement settings. Reproduces the multi-qubit tomography simulations presented in the paper.

### `km_channel_tomography.py`

Maximum-likelihood quantum tomography based on iterative Kubo–Mori (KM) updates for general quantum channels. Demonstrates reconstruction from system, environment, and correlation data in a Stinespring framework.

### `plot_convergence.py`

Utility for loading saved simulation data and generating convergence plots, including individual and mean infidelity trajectories.

### `petz_monotonicity_counterexample.py`

Searches for explicit examples in which a Petz update decreases the generalized log-likelihood. The search is performed over CPTP channels parameterized by Kraus operators and refined using Powell optimization.

### `km_monotonicity_stress_test.py`

Large-scale random search and local refinement procedure designed to identify potential violations of KM monotonicity. No violations were found in the tested parameter ranges.

### `km_vs_petz_monotonicity_comparison.py`

Direct comparison of Petz and KM updates over randomly generated quantum channels. Records likelihood changes under both updates and searches for cases where Petz decreases the likelihood while KM increases it.

## Requirements

Python 3.11+ is recommended.

Required packages:

```bash
pip install numpy scipy joblib matplotlib
```

## Reproducibility

The scripts use fixed random seeds and can be run directly:

```bash
python petz_pauli_tomography.py
python km_channel_tomography.py
python plot_convergence.py

python petz_monotonicity_counterexample.py
python km_monotonicity_stress_test.py
python km_vs_petz_monotonicity_comparison.py
```

The scripts reproduce the numerical experiments reported in the paper and generate the corresponding reconstruction and monotonicity diagnostics.

## Citation

If you use this software in academic work, please cite:

> S. Murk et al.,
> *Connecting Quantum Tomography and Quantum Retrodiction*.

A BibTeX entry and DOI will be added upon publication.

## License

This project is released under the MIT License. You are free to use, modify, and distribute the code, provided that the original copyright notice and license text are retained.

If you use this software in academic work, please cite the accompanying paper.
