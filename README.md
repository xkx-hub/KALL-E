# KALL-E

![System Overview](./figures/kalle-architecture.jpg)

## Overview

This repository contains the inference utilities for **KALL-E**, a text-to-speech
system that predicts continuous speech representations using a single
autoregressive language model.

## Key Features

- **Autoregressive Language Modeling**: Utilizes an autoregressive approach for next-distribution prediction in text-to-speech synthesis.
- **Continuous Speech Distribution**: Directly models and predicts continuous speech distributions conditioned on text, avoiding reliance on diffusion-based components.
- **FlowVAE**: Employs FlowVAE to extract continuous speech distributions from waveforms, rather than using discrete speech tokens.
- **Single AR Language Model**: Uses a single autoregressive language model to predict continuous speech distributions from text, constrained by Kullback-Leibler divergence loss.
- **Simplified Paradigm**: Offers a more straightforward and effective approach for using continuous speech representations in TTS.

## Environment Setup

- Python 3.8 or higher
- PyTorch with CUDA support
- Transformers
- NumPy
- SciPy
- alias-free-torch

Install the dependencies with:
```bash
pip install torch transformers numpy scipy alias-free_torch
```
