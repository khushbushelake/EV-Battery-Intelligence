
# ==================================================
# EV BATTERY INTELLIGENCE SYSTEM
# Data Preparation
# ==================================================

import numpy as np
from scipy.io import loadmat
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Original training dataset
DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "Train"
    / "TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat"
)

# Load data
print("Loading battery dataset...")
data = loadmat(DATA_PATH)

X = data["X"].T.astype(np.float32)
Y = data["Y"].flatten().astype(np.float32)

print("\nOriginal data shape:")
print("X:", X.shape)
print("Y:", Y.shape)

# Check SOC values outside the expected range
print("\nSOC quality check:")
print("SOC below 0:", np.sum(Y < 0))
print("SOC above 1:", np.sum(Y > 1))

# We retain these 50 values for now.
# We will not silently clip or delete observations.

# Chronological 80:20 split
split_index = int(len(X) * 0.80)

X_train = X[:split_index]
Y_train = Y[:split_index]

X_val = X[split_index:]
Y_val = Y[split_index:]

print("\nTraining and validation split:")
print("Training X:", X_train.shape)
print("Training Y:", Y_train.shape)
print("Validation X:", X_val.shape)
print("Validation Y:", Y_val.shape)

# Save prepared data
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "prepared_data.npz"

np.savez_compressed(
    OUTPUT_PATH,
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val
)

print("\nPrepared data saved successfully!")
print("Location:", OUTPUT_PATH)
