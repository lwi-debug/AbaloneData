import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
data_path = 'abalone_data.csv'  # Update this with the correct path to your data
abalone_data = pd.read_csv(data_path)
abalone_data_numeric = abalone_data.drop(columns=['Sex'])

# Standardize the data
scaler = StandardScaler()
abalone_scaled = scaler.fit_transform(abalone_data_numeric)

# Perform PCA
pca = PCA()
pca.fit(abalone_scaled)

# Calculate percentage of variance explained by each component
pve = pca.explained_variance_ratio_ * 100  # Multiply by 100 to convert to percentage
cumulative_pve = np.cumsum(pve)

# Plotting the PVE and cumulative PVE
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(pve) + 1), pve, alpha=0.6, align='center', label='Individual explained variance')
plt.step(range(1, len(pve) + 1), cumulative_pve, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained Variance Percentage')
plt.xlabel('Principal Component')
plt.title('PVE and Cumulative PVE by Principal Component')
plt.legend(loc='best')
plt.xticks(range(1, len(pve) + 1))
plt.show()

