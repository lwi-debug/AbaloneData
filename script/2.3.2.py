import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data_path = "abalone_data.csv"  # Replace with your actual data file path
abalone_data = pd.read_csv(data_path)

# Exclude the 'Sex' column
abalone_data_numeric = abalone_data.drop(columns=['Sex'])

# Standardize the data
scaler = StandardScaler()
abalone_scaled = scaler.fit_transform(abalone_data_numeric)

# Initialize PCA
pca = PCA(n_components=2)  # We focus on the first two principal components

# Perform PCA
principal_components = pca.fit_transform(abalone_scaled)

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Analyze the PCA output
print("Explained variance by component:", pca.explained_variance_ratio_)

# Loadings of the first two principal components
loadings = pca.components_.T
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=abalone_data_numeric.columns)
print("Loading vectors for the first two principal components:")
print(loading_matrix)

# Plotting the results (optional, for visualization)
plt.scatter(pc_df['PC1'], pc_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Abalone Dataset')
plt.grid(True)
plt.show()
