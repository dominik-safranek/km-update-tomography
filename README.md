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

## Citation

If you use this software in academic work, please cite:

> D. Šafránek *et al.*,
> *Connecting Quantum Tomography and Quantum Retrodiction*.

A BibTeX entry and DOI will be added upon publication.
