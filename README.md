# UNSW-NB15 Cyber Intrusion Detection

This repository provides a PyTorch-based pipeline for training and evaluating neural network models on the UNSW-NB15 dataset, a widely used benchmark for network intrusion detection research.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project-Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Customization](#customization)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Overview

This project implements a machine learning pipeline for binary classification of network traffic as normal or attack, using the UNSW-NB15 dataset. It includes:
- Data loading and preprocessing (automatic handling of categorical and numerical features)
- Model training and evaluation (PyTorch)
- Utility scripts for reproducibility and scaling

---

## Dataset

The project uses the following files from the [UNSW-NB15 dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/):
- `UNSW_NB15_training-set.csv` (main training data)
- `UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, `UNSW-NB15_4.csv` (additional data)
- `NUSW-NB15_features.csv` (feature descriptions)
- `UNSW-NB15_LIST_EVENTS.csv` (event list)

**Note:** The data directory should contain all the above files for full training.

---

## Project Structure

```
.
├── data/                # Dataset CSV files
├── ipynb/               # Jupyter notebooks and saved models
├── train.py             # Main training and evaluation script
├── prac.py              # Additional experiments or utilities
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker container specification
├── README.md            # Project documentation
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd UNSW-NB15
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare the data

Ensure all required CSV files are in the `data/` directory.

### 2. Train and evaluate the model

```bash
python train.py
```

- The script will automatically:
  - Load and concatenate all data files
  - Encode categorical features
  - Standardize numerical features
  - Train a neural network
  - Print evaluation metrics (accuracy, precision, recall, F1 score)

### 3. Jupyter Notebooks

Example notebooks are available in the `ipynb/` directory for interactive exploration and experimentation.

---

## Docker

To run the project in a containerized environment:

```bash
docker build -t unsw-nb15-cyber .
docker run --rm -v $(pwd)/data:/app/data unsw-nb15-cyber
```

- This will install dependencies and run `train.py` inside the container.
- Mount your local `data/` directory to `/app/data` in the container for data access.

---

## Customization

- **Model Architecture:** Edit the `Net` class in `train.py` to change the neural network structure.
- **Training Parameters:** Adjust epochs, batch size, or learning rate directly in `train.py`.
- **Data Preprocessing:** The script automatically detects and encodes categorical columns. You can modify this logic in the `load_data()` function.

---

## Results

- Model weights are saved as `model.pth` in the `ipynb/` directory after training.
- Evaluation metrics are printed to the console after training.

---

## References

- [UNSW-NB15 Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- [PyTorch Documentation](https://pytorch.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

## License

This project is for academic and research purposes. Please refer to the UNSW-NB15 dataset license for data usage terms.
