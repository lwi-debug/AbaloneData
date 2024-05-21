import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from adjustText import adjust_text

# Load and prepare data
data_path = 'abalone_data.csv'  # Update with your data path
abalone_data = pd.read_csv(data_path)
abalone_data_numeric = abalone_data.drop(columns=['Sex'])

# Standardizing the data
scaler = StandardScaler()
abalone_scaled = scaler.fit_transform(abalone_data_numeric)

# PCA
pca = PCA(n_components=8)  # Using two principal components
pca_result = pca.fit_transform(abalone_scaled)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Plotting the biplot
plt.figure(figsize=(14, 10))  # Further increased figure size
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, s=100)  # Larger marker size

texts = []
for i, var_names in enumerate(abalone_data_numeric.columns):
    arrow = plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', alpha=0.9, lw=3, head_width=0.05)  # Thicker arrows
    texts.append(plt.text(loadings[i, 0]*1.2, loadings[i, 1]*1.2, var_names, color='blue', ha='center', va='center', fontsize=14))

plt.xlabel("PC1", fontsize=16)
plt.ylabel("PC2", fontsize=16)
plt.title("PCA Biplot with Loading Vectors", fontsize=18)
plt.grid(True)

# Set specific axis limits as requested
plt.xlim(-1, 2)
plt.ylim(-1, 1)

# Adjust text to prevent overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='blue', lw=0.5))

plt.show()
