import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Calcul de l'âge attendu

# Sélectionner toutes les variables nécessaires
numeric_cols = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
data = data[numeric_cols]

# Standardisation des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Appliquer K-means avec k clusters
k = 4  # Par exemple, nous choisissons 4 clusters
kmeans = KMeans(n_clusters=k, random_state=0).fit(data_scaled)
data['Cluster'] = kmeans.labels_

# Réduire les dimensions à 3 composants principaux pour la visualisation
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(data_scaled)
data['PC1'] = principal_components[:, 0]
data['PC2'] = principal_components[:, 1]
data['PC3'] = principal_components[:, 2]

# Création du graphique 3D
fig = px.scatter_3d(data, x='PC1', y='PC2', z='PC3',
                    color='Cluster',  # Colorer par cluster
                    title='Clustering des ormeaux avec K-means',
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
                    size_max=5)

# Personnaliser le graphique
fig.update_traces(marker=dict(size=5))

# Initial camera position
initial_camera = dict(up=dict(x=0, y=0, z=-1),
                      center=dict(x=0, y=0, z=0),
                      eye=dict(x=-1, y=-2, z=2))

fig.update_layout(scene_camera=initial_camera, title=dict(x=0.5))

# Afficher le graphique
fig.show()
