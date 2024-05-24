import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
pve = np.round(pca.explained_variance_ratio_ * 100, 2)  # Multiply by 100 to convert to percentage
cumulative_pve = np.round(np.cumsum(pve), 2)

# Create a subplot with 1 row and 2 columns
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.5],  # Adjust row heights
    subplot_titles=("PVE and Cumulative PVE by Principal Component", "PVE Values"),
    specs=[[{"type": "bar"}], [{"type": "table"}]],
    vertical_spacing=0.20  # Adjust vertical spacing
)

# Add individual explained variance as bars
fig.add_trace(go.Bar(
    x=list(range(1, len(pve) + 1)),
    y=pve,
    name='Individual Explained Variance',
    marker=dict(color='rgba(140, 28, 19, 0.6)')  # Custom color
), row=1, col=1)

# Add cumulative explained variance as a step line
fig.add_trace(go.Scatter(
    x=list(range(1, len(pve) + 1)),
    y=cumulative_pve,
    mode='lines+markers',
    name='Cumulative Explained Variance',
    line=dict(color='rgb(191, 67, 66)'),
    marker=dict(symbol='circle')
), row=1, col=1)

# Add table with PVE values
fig.add_trace(go.Table(
    header=dict(values=["Principal Component", "Individual PVE (%)", "Cumulative PVE (%)"],
                fill_color='rgb(231, 215, 193)',
                align='left'),
    cells=dict(values=[list(range(1, len(pve) + 1)), pve, cumulative_pve],
               fill_color='rgba(231, 215, 193, 0.5)',
               align='left')
), row=2, col=1)

# Customize layout
fig.update_layout(
    height=1000,  # Increase height to make space for the table
    showlegend=True,
    bargap=0.5,
    margin=dict(t=100, b=100)  # Adjust top and bottom margins for more space
)

# Show plot
fig.show()
