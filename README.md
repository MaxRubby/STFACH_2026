# Modeling Continuous and Heterogeneous Spatio-Temporal Dependencies for Accurate Traffic Forecasting


## 📦 Overview

This repository contains the **official implementation** of STFACH (Spatio-Temporal Forecasting with Adaptive Continuous Heterogeneous modeling).

🔗 **Zenodo DOI:** [10.5281/zenodo.18303795](https://zenodo.org/records/18303795)

The repository includes:
- `model/`: Model architecture and implementation details
- `data/`: Dataset preparation
- `ckpt/`: Pre-trained checkpoints for evaluation on corresponding datasets
- `lib/`: Utility functions and metrics
- `finetune_his/`: Fine-tuning weights

## Citation
Please cite this paper as follows if it is helpful to your work:
```
M. Wen, X. Niu, J. Zhu and P. Fournier-Viger, "Modeling Continuous and Heterogeneous Spatio-Temporal Dependencies for Accurate Traffic Forecasting," in IEEE Internet of Things Journal, vol. 13, no. 8, pp. 17696-17715, 15 April15, 2026, doi: 10.1109/JIOT.2026.3663402.
keywords: {Adaptation models;Forecasting;Predictive models;Internet of Things;Mathematical models;Correlation;Semantics;Roads;Accuracy;Vectors;Graph neural network (GNN);intelligent transportation system (ITS);neural differential equations (NDEs);spatio-temporal dependencies;traffic forecasting},
```
Link: https://ieeexplore.ieee.org/document/11389793

---

## Data Preparation

We utilize datasets from [LibCity](https://github.com/LibCity/Bigscity-LibCity), a standardized and comprehensive benchmark for urban spatio-temporal data mining. Please download the required datasets and place them in the corresponding folders under `data/`.

### Supported Datasets

**Highway Traffic Flow:**
- PEMS03
- PEMS04
- PEMS07
- PEMS08

**Highway Traffic Speed:**
- PEMS-BAY
- METR-LA

**Urban Mobility Demand:**
- NYCTAXI (NYC-TAXI)
- BIKECHI (CHI-BIKE)

**Metro Crowd Flow:**
- HZMETRO
- SHMETRO

## Requirements

The implementation requires Python 3.9 or higher. Install the dependencies using:

```bash
pip install -r req.txt
```

### Key Dependencies

- PyTorch (with CUDA support recommended)
- scipy
- pandas
- tensorboard
- scikit-learn
- torchdiffeq
- fastdtw==0.3.4
- networkx
- matplotlib
- seaborn

For a complete list of dependencies, please refer to `req.txt`.

## Usage

### Training

To train the model on a specific dataset:

```bash
python train.py -d HZMETRO -m train -e 1 -c comment
```

**Arguments:**
- `-d`: Dataset name (e.g., HZMETRO, PEMS04, METR_LA)
- `-m`: Mode (train or test)
- `-e`: Experiment ID
- `-c`: Comment or description for the experiment

### Testing

To evaluate a pre-trained model:

```bash
python train.py -d HZMETRO -m test -e 1 -c comment -cont 95
```

**Additional Arguments:**
- `-cont`: Checkpoint epoch number to load for evaluation

## Acknowledgments

We acknowledge the LibCity project for providing standardized datasets and evaluation protocols for traffic forecasting research.

---
