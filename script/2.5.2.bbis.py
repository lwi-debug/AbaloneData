import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import numpy as np

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Calcul de l'âge à partir des anneaux

def remove_outliers(df, column_list):
    for column in column_list:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df

# Appliquer la fonction de suppression des outliers
numeric_cols = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
data = remove_outliers(data, numeric_cols)

# Standardisation des données
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Définition des prédicteurs et de la variable cible
X = data[numeric_cols]  # Exclure 'Rings' et 'Sex'
y = data['Age']

# Ajout de la constante pour l'intercept
X = sm.add_constant(X)

# Création du modèle de régression linéaire multiple
model = sm.OLS(y, X)

# Ajustement du modèle
results = model.fit()

# Affichage du résumé des résultats du modèle
print(results.summary())

# Créer le graphique des coefficients
coefficients = results.params[1:]  # Exclure la constante
features = numeric_cols

fig = go.Figure()

fig.add_trace(go.Bar(
    x=features,
    y=coefficients,
    text=[f"{coef:.4f}" for coef in coefficients],
    textposition='outside',
    marker_color=['#F7B267', '#F79D65', '#F4845F', '#F27059', '#F7B267', '#F79D65', '#F4845F'],
    name='Coefficients'
))

# Personnaliser le graphique
fig.update_layout(
    title='Coefficients de la régression linéaire multiple',
    xaxis=dict(title='Caractéristiques'),
    yaxis=dict(title='Valeur des coefficients'),
    showlegend=False
)

# Afficher le graphique
fig.show()

# Plot 3D de la régression linéaire
fig_3d = go.Figure()

# Choisir deux variables explicatives pour le plot 3D
x_var = 'Shell weight'
y_var = 'Diameter'

# Surface de régression
x_range = np.linspace(data[x_var].min(), data[x_var].max(), 100)
y_range = np.linspace(data[y_var].min(), data[y_var].max(), 100)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
z_mesh = (results.params[0] +
          results.params[numeric_cols.index(x_var) + 1] * x_mesh +
          results.params[numeric_cols.index(y_var) + 1] * y_mesh)

# Ajouter les points de données
fig_3d.add_trace(go.Scatter3d(
    x=data[x_var], y=data[y_var], z=data['Age'],
    mode='markers',
    marker=dict(size=4, color=data['Age'], colorscale='Viridis', opacity=0.8)
))

# Ajouter la surface de régression
fig_3d.add_trace(go.Surface(
    x=x_range, y=y_range, z=z_mesh,
    colorscale='Viridis', opacity=0.5
))

# Personnaliser le graphique 3D
fig_3d.update_layout(
    title='Régression Linéaire Multiple 3D',
    scene=dict(
        xaxis_title=x_var,
        yaxis_title=y_var,
        zaxis_title='Age'
    )
)

# Afficher le graphique 3D
fig_3d.show()
