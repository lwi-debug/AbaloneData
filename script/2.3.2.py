import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load the data
data_path = "abalone_data.csv"
abalone_data = pd.read_csv(data_path)

# Exclude the 'Sex' column and prepare for PCA
abalone_data_numeric = abalone_data.drop(columns=['Sex'])

# Standardize the data
scaler = StandardScaler()
abalone_scaled = scaler.fit_transform(abalone_data_numeric)

# Initialize and perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(abalone_scaled)

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add 'Whole weight' and 'Diameter' back for coloring and sizing
pc_df['Whole weight'] = abalone_data['Whole weight']
pc_df['Diameter'] = abalone_data['Diameter']

# Plotting with Plotly
fig = px.scatter(pc_df, x='PC1', y='PC2',
                 color='Whole weight',
                 size='Diameter',
                 color_continuous_scale=['#fcba04', '#a50104'],  # Gradient from light orange to deep red
                 title='PCA of Abalone Dataset',
                 labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                 size_max=10)

# Remove the white circle borders around points
fig.update_traces(marker=dict(line=dict(width=0)))

# Show plot
fig.show()
