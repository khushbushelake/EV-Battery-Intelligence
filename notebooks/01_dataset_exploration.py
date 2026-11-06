
# ==========================================
# EV BATTERY INTELLIGENCE SYSTEM
# Step 1: Dataset Exploration
# ==========================================

# Import required libraries
import numpy as np
from scipy.io import loadmat
from pathlib import Path

# Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Specify the training dataset
dataset_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "Train"
    / "TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat"
)

# Load the MATLAB dataset
data = loadmat(dataset_path)

# Extract input features and target values
X = data["X"]
Y = data["Y"]

# Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("Dataset file:", dataset_path.name)
print("Input features (X) shape:", X.shape)
print("Target values (Y) shape:", Y.shape)

print("\n========== FEATURE STATISTICS ==========")

for i in range(X.shape[0]):
    print(
        f"Feature {i + 1}: "
        f"Min = {np.min(X[i]):.4f}, "
        f"Max = {np.max(X[i]):.4f}, "
        f"Mean = {np.mean(X[i]):.4f}"
    )

print("\n========== TARGET STATISTICS ==========")

print("Minimum SOC:", np.min(Y))
print("Maximum SOC:", np.max(Y))
print("Average SOC:", np.mean(Y))

print("\nDataset inspection completed successfully!")

# ==========================================
# Step 2: Visualize SOC Distribution
# ==========================================

import matplotlib.pyplot as plt

# Convert SOC array into a one-dimensional array
soc_values = Y.flatten()

# Create histogram
plt.figure(figsize=(10, 5))

plt.hist(soc_values, bins=50, edgecolor="black")

# Add chart details
plt.title("Distribution of Battery State of Charge (SOC)")
plt.xlabel("State of Charge (Normalized)")
plt.ylabel("Number of Observations")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Save the graph in the reports folder
graph_path = PROJECT_DIR / "reports" / "soc_distribution.png"
plt.savefig(graph_path, dpi=300)

print("\nSOC distribution graph saved at:")
print(graph_path)

# Display the graph
plt.show()

# ==========================================
# Step 3: Feature Distribution Analysis
# ==========================================

print("\n========== SOC DATA QUALITY ==========")

total = len(soc_values)

below_zero = np.sum(soc_values < 0)
above_one = np.sum(soc_values > 1)
high_soc = np.sum(soc_values >= 0.95)

print("Total SOC observations:", total)
print(f"SOC below 0: {below_zero} ({below_zero / total * 100:.4f}%)")
print(f"SOC above 1: {above_one} ({above_one / total * 100:.4f}%)")
print(f"SOC at or above 0.95: {high_soc} ({high_soc / total * 100:.2f}%)")


# ==========================================
# Step 4: Visualize All Five Input Features
# ==========================================
fig, axes = plt.subplots(
    5, 1,
    figsize=(12, 18),
    sharex=True,
    constrained_layout=True
)
for i in range(5):
    axes[i].hist(X[i], bins=40, edgecolor="black")
    axes[i].set_title(f"Distribution of Feature {i + 1}")
    axes[i].set_ylabel("Frequency")
    axes[i].grid(axis="y", alpha=0.3)

axes[-1].set_xlabel("Normalized Value")

plt.tight_layout()

feature_graph = PROJECT_DIR / "reports" / "feature_distributions.png"
plt.savefig(feature_graph, dpi=300)

print("\nFeature distribution graph saved:", feature_graph)

plt.show()


# ==========================================
# Step 5: SOC Variation Over Observations
# ==========================================

sample_count = min(5000, len(soc_values))

plt.figure(figsize=(12, 5))

plt.plot(
    np.arange(sample_count),
    soc_values[:sample_count],
    linewidth=0.8
)

plt.title("Battery SOC Variation - First 5000 Observations")
plt.xlabel("Observation Index")
plt.ylabel("State of Charge (SOC)")
plt.grid(alpha=0.3)
plt.tight_layout()

soc_graph = PROJECT_DIR / "reports" / "soc_variation.png"
plt.savefig(soc_graph, dpi=300)

print("SOC variation graph saved:", soc_graph)

plt.show()

print("\nAll data analysis graphs generated successfully!")

# ==========================================
# Step 6: Investigate SOC Sequence
# ==========================================

print("\n========== SOC SEQUENCE INVESTIGATION ==========")

first_5000 = soc_values[:5000]

print("Unique SOC values in first 5000 observations:",
      len(np.unique(first_5000)))

print("\nSelected SOC values:")

for index in [0, 100, 500, 1000, 2000, 3000, 3500, 4000, 4500, 4999]:
    print(f"Index {index}: SOC = {soc_values[index]:.6f}")


# ==========================================
# Step 7: Check Missing and Invalid Values
# ==========================================

print("\n========== DATA QUALITY CHECK ==========")

print("Missing values in X:", np.isnan(X).sum())
print("Missing values in Y:", np.isnan(Y).sum())

print("Infinite values in X:", np.isinf(X).sum())
print("Infinite values in Y:", np.isinf(Y).sum())

print("SOC below 0:", np.sum(Y < 0))
print("SOC above 1:", np.sum(Y > 1))


# ==========================================
# Step 8: Check Feature Relationships
# ==========================================

print("\n========== FEATURE CORRELATIONS ==========")

correlation_data = np.vstack([X, Y])

correlation_matrix = np.corrcoef(correlation_data)

print("Correlation matrix:")
print(np.round(correlation_matrix, 3))

plt.figure(figsize=(9, 7))

plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

labels = [
    "Feature 1",
    "Feature 2",
    "Feature 3",
    "Feature 4",
    "Feature 5",
    "SOC"
]

plt.xticks(range(6), labels, rotation=45)
plt.yticks(range(6), labels)

plt.title("Feature and SOC Correlation Matrix")
plt.tight_layout()

correlation_path = PROJECT_DIR / "reports" / "correlation_matrix.png"

plt.savefig(correlation_path, dpi=300)

print("\nCorrelation graph saved:", correlation_path)

plt.show()
