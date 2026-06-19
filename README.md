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

## License

This project is licensed under the MIT License.

Copyright (c) 2026 Dominik Šafránek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
