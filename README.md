# Adaptive Quantum Semantic Communication for Resource-Efficient Dynamic Quantum Networks

**Summer Research 2026** · University of Alabama in Huntsville

## Overview

This repository implements an adaptive framework for quantum semantic communication over dynamic quantum networks, developed as part of the **Summer Research 2026** program at the University of Alabama in Huntsville. The goal is to balance semantic representation quality against limited entanglement resources by selecting an appropriate compression level in response to live network conditions.

Rather than fixing the semantic alphabet size in advance, the system observes network state—including channel fidelity, entanglement success rate, delay, distance, and traffic load—and uses a contextual multi-armed bandit (LinUCB) to choose the compression level online.

Network simulation is performed with [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe).

## Approach

| Component | Description |
|-----------|-------------|
| Semantic front-end | CIFAR-10 feature extraction with ResNet-18 and K-means clustering |
| Policy | LinUCB over discrete compression levels |
| Network backend | Two-node entanglement generation in SeQUeNCe |
| Objective | Maximize end-to-end semantic–quantum fidelity under resource constraints |
| Context features | Network observables and load-related interaction terms |
| Learning mode | Online bandit updates from episode feedback |

## Repository structure

```
adaptive_qsc/
  config.py
  semantic/          # concept construction and compression
  qnetwork/          # SeQUeNCe entanglement environment
  learning/          # LinUCB policy and reward design
  runtime/           # shared episode loop
  experiments/       # baseline, training, and evaluation scripts
  models/            # trained policy checkpoints
  results/           # experiment outputs and figures
```

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/BAZuah/adaptive-quantum-semantic-communication.git
cd adaptive-quantum-semantic-communication
pip install -r requirements.txt
```

SeQUeNCe must be installed separately and available in your Python environment. See the [SeQUeNCe documentation](https://github.com/sequence-toolbox/SeQUeNCe) for setup instructions.

## Running experiments

From the project root:

```bash
# Static baseline
python experiments/01_static_qsc_baseline.py --episodes 90

# Train LinUCB policy
python experiments/02_train_linucb.py --episodes 250

# Adaptive vs fixed-compression comparison
python experiments/03_adaptive_vs_fixed.py --episodes 150

# Generate summary figures
python results/generate_figures.py

# Additional paper-style comparison figures
python experiments/04_paper_style_figures.py
```

Outputs are written to `results/`.

## Paper results

Precomputed figures used in the Summer Research 2026 write-up are included under `results/paper_results/`:

| File prefix | Content |
|-------------|---------|
| `figA_*` | Overall FSQ: adaptive vs static |
| `figB_*` | FSQ by network tier |
| `figC_*` | Semantic size vs FSQ |
| `figD_*` | Semantic size vs resources |
| `figE_*` | Fidelity vs resources |
| `figF_*` | Compression by tier and condition |
| `figG_*` | Fs, Fc, and FSQ breakdown |
| `figH_*` | Size vs FSQ by network condition |
| `summary.json` | Numeric summary for all paper figures |

## Expected behavior

1. No single fixed compression level is optimal across all network conditions.
2. The adaptive policy increases compression under stressed network tiers and retains richer representations under favorable conditions.
3. Adaptive control improves performance relative to a naive fixed full-representation baseline, particularly under high load or degraded channel quality.

## Author

**Md Bokhtiar Al Zami**  
Ph.D. Student, University of Alabama in Huntsville  
[GitHub](https://github.com/BAZuah) · [Google Scholar](https://scholar.google.com/citations?user=NuLSqUIAAAAJ)

## License

This project is released for academic and research use. Please cite the repository if you use this code in your work.
