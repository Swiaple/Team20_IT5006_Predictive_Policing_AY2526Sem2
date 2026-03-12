# XGBoost Crime Forecasting

This folder contains the XGBoost implementation for short-term spatiotemporal crime forecasting in Chicago.

## Task
The model uses the past 30 days of crime observations to predict the crime count for the corresponding time slot on the next day.

## Crime Categories
The model is trained separately for the following five crime categories:

- THEFT
- BATTERY
- CRIMINAL_DAMAGE
- ASSAULT
- DECEPTIVE_PRACTICE

## Data Input
The code expects the processed data to be stored under the following directory:

```text
processed/
├── tensor.npy
├── time_features.npy
└── meta.json
