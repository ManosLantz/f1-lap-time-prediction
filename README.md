F1 Lap Time Prediction

This project builds a machine learning pipeline that predicts the next Formula 1 lap time using lap-level timing and race information from FastF1.
The goal is to model realistic race pace using only publicly available timing data.

The system processes multiple seasons (2022–2025), extracts structured features, and evaluates performance on unseen 2025 races to simulate real predictive conditions.

Overview

The project consists of three main parts:

1. Data Extraction & Cleaning

Loads race sessions using FastF1

Removes pit laps, SC/VSC/Yellow laps, wet tyres, and inconsistent timing

Merges weather data

Computes additional race and track metadata (lap numbers, track characteristics, etc.)

2. Feature Engineering

The pipeline generates a set of features describing:

Lap timing and sector performance

Tyre behaviour and degradation

Race progression

Driver intent signals

Track physics (e.g., aero vs power characteristics)

Basic car performance indicators

More features may be added or refined later as the modelling develops.

3. Modelling

Uses XGBoost to predict the next lap time.
A custom tuning script evaluates hyperparameters directly on unseen 2025 races.
A naive baseline (“next lap = previous lap”) is also included for comparison.

Current model performance:

MAE ≈ 0.45 s

RMSE ≈ 0.65 s

~20% improvement over the naive baseline

These values may change as additional features or models are added.

Project Structure
config.py            # Configuration and race info
f1_data.py           # Dataset creation and FastF1 processing
features.py          # Feature engineering
train_model.py       # Training, evaluation, baselines
tune_xgb_direct.py   # Direct hyperparameter tuning
tune_xgboost.py      # Alternative tuning approach (for comparison)
utils.py             # Helper functions


The FastF1 cache and generated datasets are ignored by Git.

Future Extensions

Possible directions include:

Improved modelling of tyre degradation

Additional driver-state or ERS-related features

Testing alternative ML models or architectures

Extending the dataset to more seasons or circuits

Notes

This is an ongoing project and the README will expand as the work progresses.
FastF1 must be installed and properly cached locally to reproduce the data extraction steps.
