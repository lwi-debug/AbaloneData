import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Calcul de l'âge attendu

# Sélectionner toutes les variables nécessaires
numeric_cols = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
data = data[numeric_cols + ['Age']]

# Standardisation des données
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Définition des prédicteurs et de la variable cible
X = data[numeric_cols]
y = data['Age']

# Ajout d'une constante (intercept)
X = sm.add_constant(X)

# Ajustement du modèle de régression linéaire
model = sm.OLS(y, X).fit()

# Effectuer la prédiction sur l'ensemble des données
data['Predicted Age'] = model.predict(X)

# Comparaison de l'âge prédit avec l'âge attendu
data['Age Difference'] = data['Predicted Age'] - data['Age']

# Calculer des métriques statistiques
mae = np.mean(np.abs(data['Age Difference']))
print(f"Erreur Moyenne Absolue (MAE): {mae:.2f}")

# Calcul du coefficient de détermination R^2
r_squared = model.rsquared
print(f"Coefficient de détermination (R^2): {r_squared:.2f}")

# Création du graphique de comparaison 3D
fig = go.Figure()

# Ajouter les âges réels en 3D
fig.add_trace(go.Scatter3d(
    x=data.index,  # Indices des échantillons
    y=data['Age'],
    z=data['Predicted Age'],
    mode='markers',
    marker=dict(size=3, color='#FFBF00'),
    name='Âge Réel'
))

# Ajouter les âges prédits en 3D
fig.add_trace(go.Scatter3d(
    x=data.index,  # Indices des échantillons
    y=data['Predicted Age'],
    z=data['Age'],
    mode='markers',
    marker=dict(size=3, color='#E83F6F'),
    name='Âge Prédit'
))

# Personnaliser le graphique
fig.update_layout(
    title='Comparaison des âges réels et prédits des ormeaux (3D)',
    scene=dict(
        xaxis=dict(title='Échantillons'),
        yaxis=dict(title='Âge Réel'),
        zaxis=dict(title='Âge Prédit')
    ),
    showlegend=True
)

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
