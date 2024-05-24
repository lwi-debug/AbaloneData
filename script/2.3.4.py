import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Charger les données
data_path = "abalone_data.csv"  # Remplacer par le chemin de votre fichier de données
abalone_data = pd.read_csv(data_path)

# Exclure la colonne 'Sex' et préparer pour la PCA
abalone_data_numeric = abalone_data.drop(columns=['Sex'])

# Standardiser les données
scaler = StandardScaler()
abalone_scaled = scaler.fit_transform(abalone_data_numeric)

# Initialiser et réaliser la PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(abalone_scaled)

# Créer un DataFrame avec les composantes principales
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Ajouter les vecteurs de chargement pour les variables originales
loadings = pca.components_.T

# Palette de couleurs
colors = ['#6A040F', '#9D0208', '#D00000', '#DC2F02', '#E85D04', '#F48C06', '#FAA307', '#FFBA08']

# Créer des sous-graphiques avec Plotly
fig = make_subplots(rows=1, cols=2, subplot_titles=("Intervalle -1 à 1", "Intervalle -5 à 5"))

# Ajouter les scores des composantes principales pour le premier graphique
fig.add_trace(go.Scatter(x=pc_df['PC1'], y=pc_df['PC2'],
                         mode='markers',
                         marker=dict(size=6, color='#7209b7'),  # Rouge tomate
                         name='Scores des Composantes Principales'), row=1, col=1)

# Ajouter les vecteurs de chargement pour le premier graphique
for i, (var, color) in enumerate(zip(abalone_data_numeric.columns, colors)):
    fig.add_trace(go.Scatter(x=[0, loadings[i, 0]], y=[0, loadings[i, 1]],
                             mode='lines+text',
                             text=[None, var],
                             textposition='top center',
                             textfont=dict(size=14),  # Taille du texte
                             name=var,
                             line=dict(color=color, width=2)), row=1, col=1)

# Ajouter le cercle de corrélation pour le premier graphique
fig.add_trace(go.Scatter(x=np.cos(np.linspace(0, 2 * np.pi, 100)), y=np.sin(np.linspace(0, 2 * np.pi, 100)),
                         mode='lines',
                         name='Cercle de Corrélation',
                         line=dict(color='grey', dash='dash')), row=1, col=1)

# Ajouter les scores des composantes principales pour le deuxième graphique
fig.add_trace(go.Scatter(x=pc_df['PC1'], y=pc_df['PC2'],
                         mode='markers',
                         marker=dict(size=6, color='#7209b7'),  # Rouge tomate
                         showlegend=False), row=1, col=2)

# Ajouter les vecteurs de chargement pour le deuxième graphique
for i, (var, color) in enumerate(zip(abalone_data_numeric.columns, colors)):
    fig.add_trace(go.Scatter(x=[0, loadings[i, 0]], y=[0, loadings[i, 1]],
                             mode='lines',
                             showlegend=False,
                             line=dict(color=color, width=2)), row=1, col=2)

# Ajouter le cercle de corrélation pour le deuxième graphique
fig.add_trace(go.Scatter(x=np.cos(np.linspace(0, 2 * np.pi, 100)), y=np.sin(np.linspace(0, 2 * np.pi, 100)),
                         mode='lines',
                         showlegend=False,
                         line=dict(color='grey', dash='dash')), row=1, col=2)

# Mettre à jour la mise en page du graphique
fig.update_layout(title='Biplot avec Cercle de Corrélation',
                  showlegend=True,
                  legend=dict(x=1.05, y=1.0, traceorder="normal"))  # Position de la légende

# Mettre à jour les axes pour le premier graphique
fig.update_xaxes(title_text='Composante Principale 1', range=[-1, 1], row=1, col=1)
fig.update_yaxes(title_text='Composante Principale 2', range=[-1, 1], row=1, col=1)

# Mettre à jour les axes pour le deuxième graphique
fig.update_xaxes(title_text='Composante Principale 1', range=[-5, 5], row=1, col=2)
fig.update_yaxes(title_text='Composante Principale 2', range=[-5, 5], row=1, col=2)

# Afficher le graphique
fig.show()
