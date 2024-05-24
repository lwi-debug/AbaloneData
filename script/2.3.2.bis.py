import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Load the data
data_path = "abalone_data.csv"  # Replace with your actual data file path
abalone_data = pd.read_csv(data_path)

# Exclude the 'Sex' column and prepare for PCA
abalone_data_numeric = abalone_data.drop(columns=['Sex'])

# Standardize the data
scaler = StandardScaler()
abalone_scaled = scaler.fit_transform(abalone_data_numeric)

# Function to remove outliers
def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Remove outliers from the whole weight column
abalone_data_numeric = remove_outliers(abalone_data_numeric, 'Whole weight')

# Standardize the data again after removing outliers
abalone_scaled = scaler.fit_transform(abalone_data_numeric)

# Initialize and perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(abalone_scaled)

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

# Exclude values where PC3 > 5
pc_df = pc_df[pc_df['PC3'] <= 5]

# Add 'Whole weight' back for coloring
pc_df['Whole weight'] = abalone_data.loc[abalone_data_numeric.index, 'Whole weight'].iloc[pc_df.index]

# Plotting with Plotly
fig = px.scatter_3d(pc_df, x='PC1', y='PC2', z='PC3',
                    color='Whole weight',
                    color_continuous_scale=['#FB8B24', '#D90368', '#820263'],  # Gradient of the specified colors
                    title='3D PCA of Abalone Dataset',
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
                    size_max=5)  # Reduce the size of the bubbles

# Reduce the size of the markers
fig.update_traces(marker=dict(size=4))

# Initial camera position
initial_camera = dict(up=dict(x=0, y=0, z=-1),
                      center=dict(x=0, y=0, z=0),
                      eye=dict(x=-1, y=-2, z=2))

fig.update_layout(scene_camera=initial_camera, title=dict(x=0.5))

# Create frames for the figure-eight path with position indicator
frames = []
num_steps = 2880  # Total number of frames for a smoother animation
for t in range(num_steps):
    angle = np.pi * 2 * (t / num_steps)  # Angle in radians
    x_eye = 1.5 * np.sin(angle) * np.cos(angle)  # Increased amplitude for larger figure-eight
    y_eye = 1.5 * np.sin(angle) * np.sin(angle)  # Increased amplitude for larger figure-eight
    z_eye = -2 * np.cos(angle)  # Increased amplitude for larger figure-eight
    camera = dict(up=dict(x=0, y=0, z=0),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=x_eye, y=y_eye, z=z_eye))

    # Add a frame with updated camera position and annotation for x, y, z
    frames.append(go.Frame(layout=dict(scene_camera=camera,
                                       annotations=[dict(
                                           x=0.5,
                                           y=0.95,
                                           xref="paper",
                                           yref="paper",
                                           text=f"Position: x={x_eye:.2f}, y={y_eye:.2f}, z={z_eye:.2f}",
                                           showarrow=False
                                       )])))

fig.frames = frames

# Create the animation settings
animation_settings = dict(frame=dict(duration=10, redraw=True), fromcurrent=True, mode='immediate')  # Slower animation with more frames per second

# Add the initial frame with the starting camera position
initial_frame = go.Frame(layout=dict(scene_camera=initial_camera,
                                     annotations=[dict(
                                         x=0.5,
                                         y=0.95,
                                         xref="paper",
                                         yref="paper",
                                         text=f"Position: x=1.00, y=1.00, z=1.00",
                                         showarrow=False
                                     )]))
fig.frames = [initial_frame] + frames

fig.update_layout(updatemenus=[dict(type='buttons',
                                    buttons=[dict(label='Play',
                                                  method='animate',
                                                  args=[None, animation_settings])])])

# Show plot
fig.show()
