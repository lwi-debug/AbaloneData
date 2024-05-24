import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Charger les données
data_path = "abalone_data.csv"
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

# Création du graphique de comparaison
fig = go.Figure()

# Ajouter les âges réels
fig.add_trace(go.Scatter(
    x=data.index,  # Indices des échantillons
    y=data['Age'],
    mode='markers',
    marker=dict(color='#FFBF00'),
    name='Âge Réel'
))

# Ajouter les âges prédits
fig.add_trace(go.Scatter(
    x=data.index,  # Indices des échantillons
    y=data['Predicted Age'],
    mode='markers',
    marker=dict(color='#E83F6F'),
    name='Âge Prédit'
))

# Personnaliser le graphique
fig.update_layout(
    title='Comparaison des âges réels et prédits des ormeaux',
    xaxis=dict(title='Échantillons'),
    yaxis=dict(title='Âge'),
    showlegend=True
)

# Afficher le graphique
fig.show()
