# Deep Learning Pipeline for Histopathology Image Classification

## Overview

This repository implements a modular deep learning pipeline for histopathology image classification.
It is built using **PyTorch Lightning** for training and **Hydra** for configuration management.

The project supports multiple training approaches:

* **Full fine-tuning**
* **Parameter-efficient fine-tuning (LoRA)**
* **Domain-Adversarial Training (DANN)**

Additional features include:

* Advanced data augmentation (Albumentations)
* Stain normalization for histopathology images
* Exponential Moving Average (EMA) of model weights
* Experiment tracking with MLflow

---

## Project Structure

```
.
├── configs/                # Hydra configuration files
│   ├── dataset/
│   ├── model/
│   ├── training/
│   └── augmentation/
│
├── src/
│   ├── data/              # Data loading, preprocessing & augmentations
│   ├── light/             # PyTorch Lightning modules
│   ├── models/            # Model architectures
│   │
│   ├── main.py            # Training entry point
│   ├── prediction.py      # Test inference script
│   ├── test_time_augmentation.py # Test time augmentation/normalization after training 
│   └── utilities.py
```

---

## Key Components

### Data Pipeline

* Custom `DatasetAugmentation` for HDF5 datasets (via `fsspec`)
* Efficient caching mechanism for remote data access
* Rich augmentation pipeline using **Albumentations**
* Optional **histopathology stain normalization** (Macenko / Reinhard)

### Models

* Transformer-based backbones:

  * DINOv2
  * Phikon
* Modular design:

  * Backbone + classification head
* Supported training strategies:

  * **LoRA** (parameter-efficient fine-tuning)
  * **Full fine-tuning**
  * **DANN** (domain adaptation via gradient reversal)

### Training (PyTorch Lightning)

* Clean separation of training logic via Lightning modules
* Features:

  * EMA (Exponential Moving Average)
  * Cosine learning rate scheduling
  * Layer-wise learning rates (for fine-tuning)
  * Multi-loss training (DANN)

---

## Configuration (Hydra)

All experiments are driven by Hydra configs:

* `configs/model/` → model architecture & method (LoRA, DANN, etc.)
* `configs/dataset/` → dataset paths, batch size, workers
* `configs/training/` → optimizer, scheduler, EMA, etc.
* `configs/augmentation/` → data augmentation settings

You can override any parameter from the command line.

---

## Installation

Create your environment with uv:

```
git clone https://github.com/pov-sam/Project-MVA-DLMI 
cd Project-MVA-DLMI 
uv sync
```

---

## Usage

### Training

```
uv run src/main.py
```

With overrides:

```
uv run src/main.py models.method=LoRA training.lr=1e-4
```

Some predefined .yaml configurations are provided for specific training setups. For instance DANN training with full augmentation:
```
uv run src/main.py augmentation=Augmentation_only models=dann_dino training=training_dann
```
---

### Prediction

```
uv run src/prediction.py
```

> Note: the checkpoint path and model definition are hardcoded in the script.

---

### Test-Time Augmentation/Normalization (TTA-N)

```
python src/test_time_augmentation.py
```
> Note: the checkpoint path, model definition, and choice of augmentation/normalization are hardcoded in the script.
---

## Design Choices

* **Hydra + Lightning**: clean experiment management and modular training
* **HDF5 + fsspec**: scalable data loading from remote storage

---

## Summary

This project provides a flexible and extensible framework for deep learning on histopathology data, combining:

* modern architectures (transformers),
* efficient fine-tuning strategies,
* and robust training techniques.

The modular design allows easy experimentation and extension to other datasets or tasks.




