# Dissertation Project

Machine learning and deep learning pipeline for the classification, localisation, and change detection of transient signals in power cable systems.

## Project structure

- `src/dissertation_project/` reusable Python code
- `scripts/` command-line scripts for running experiments and generating figures
- `notebooks/` exploratory analysis only
- `config/` experiment parameters
- `reports/figures/` final thesis figures
- `artifacts/` generated models, predictions, logs, ignored by Git
- `data/` raw and processed datasets, ignored by Git

## Experiments

1. Experiment 1: partial-discharge-like source localisation
2. Experiment 1: voltage-invariant evaluation
3. Experiment 1: open-set Non-PD rejection
4. Experiment 1: noise robustness
5. Experiment 2: laboratory shield integrity classification
6. Experiment 2: factory change detection

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
