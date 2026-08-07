# Adaptive Quantum Semantic Communication for Resource-Efficient Dynamic Quantum Networks

**Md Bokhtiar Al Zami** and **Dinh C. Nguyen**  
Summer Research 2026 · University of Alabama in Huntsville

**Repository:** https://github.com/BAZuah/adaptive-quantum-semantic-communication

## Overview

Quantum communication networks operate with limited entanglement resources whose availability and quality vary with channel conditions, memory decoherence, and traffic load. Existing quantum semantic communication (QSC) methods typically rely on a fixed semantic representation, which can impose excessive resource demand when the network becomes constrained.

This repository implements an **adaptive QSC framework** that adjusts the semantic representation according to the current network condition. A contextual multi-armed bandit (MAB) uses network probing, traffic load, semantic demand, and residual service capacity to select a suitable representation ratio. The chosen ratio sets the semantic granularity and the corresponding entangled-pair demand, while quantum semantic fidelity reflects both semantic preservation and quantum delivery quality.

Network simulation is performed with [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe).

## System components

| Component | Description |
|-----------|-------------|
| Semantic front-end | CIFAR-10 feature extraction with ResNet-18 and K-means clustering |
| Adaptive policy | Contextual MAB (LinUCB) over discrete representation ratios |
| Network backend | Two-node entanglement generation in SeQUeNCe |
| Context | Network probing, traffic load, semantic demand, and residual capacity |
| Objective | Balance semantic fidelity against entangled-pair consumption under dynamic conditions |

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
  results/paper_results/   # figures accompanying the paper
```

## Installation

```bash
git clone https://github.com/BAZuah/adaptive-quantum-semantic-communication.git
cd adaptive-quantum-semantic-communication
pip install -r requirements.txt
```

SeQUeNCe must be installed separately. See the [SeQUeNCe documentation](https://github.com/sequence-toolbox/SeQUeNCe) for setup instructions.

## Running experiments

From the project root:

```bash
# Static QSC baseline
python experiments/01_static_qsc_baseline.py --episodes 90

# Train adaptive MAB policy
python experiments/02_train_linucb.py --episodes 250

# Adaptive vs fixed-compression comparison
python experiments/03_adaptive_vs_fixed.py --episodes 150

# Regenerate paper figures
python experiments/06_superiority_figures.py
```

## Paper figures

Precomputed figures referenced in the paper are provided under `results/paper_results/`.

## Authors

**Md Bokhtiar Al Zami** · **Dinh C. Nguyen**  
University of Alabama in Huntsville

## License

This project is released for academic and research use. Please cite the repository if you use this code in your work.
