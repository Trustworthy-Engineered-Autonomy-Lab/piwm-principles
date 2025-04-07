# Four Principles for Physically Interpretable World Models

This repository contains the source code for this paper: https://arxiv.org/abs/2503.02143

## Prerequisites

```python
pip install -r requirements.txt
```

# Getting Started

This repository implements our framework for building **Physically Interpretable World Models**, following the principles described in our paper.

We provide scripts to collect data, train foundational models (VAE and LSTM), and run experiments for each of the three core principles.

---

## 1. Data Collection

To generate datasets for training, validation, and testing:

```bash
python dataCollect.py
```

This script collects observation-action pairs and organizes them into train, val, and test splits.

---

## 2. Baseline Training

**Train a Variational Autoencoder (VAE)**

```bash
python vae.py
```

This trains a VAE to compress high-dimensional observations into latent representations.

**Train an LSTM for Prediction**

```bash
python lstm.py
```

This LSTM model is used across all principles to perform temporal prediction in latent space.

---

## 3. Experiments for Interpretability Principles

### Principle 1: Structuring Latent Representations

To encode observations into modular latent components (e.g., physical state, image features):

```bash
python seperate_encoding.py
```

This script implements separate encoding branches for each latent subspace.

---

### Principle 2: Invariant and Equivariant Representations

To train the VAE with alignment constraints (e.g., transformations and their expected effects):

```bash
python translation_loss.py
```

This loss promotes latent invariance/equivariance aligned with physical transformations.

---

### Principle 3: Multi-Level Supervision

To incorporate mixed supervision signals during training (fully labeled, weakly labeled, and unlabeled):

```bash
python partial_supervision.py
```

This script uses weak supervision techniques (e.g., temporal smoothness) to improve interpretability.

To compare results with and without access to velocity estimation, run:

```bash
python vel_estimation.py
```

---

## Citation

If you use this repository in your work, please cite:

```bibtex
@article{sutera2025piwm,
  title={Four Principles for Physically Interpretable World Models},
  author={Sutera, A. and Mao, P. and Geng, M. and Pan, T. and Ruchkin, I.},
  journal={arXiv preprint arXiv:2503.02143},
  year={2025}
}
```

