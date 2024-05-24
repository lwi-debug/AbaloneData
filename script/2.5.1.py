import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler  # Importation manquante

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Calcul de l'âge à partir des anneaux

def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Retirer les outliers de la variable 'Shell weight'
data = remove_outliers(data, 'Shell weight')

# Standardisation des données
features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Liste pour stocker les résultats
results_list = []

# Fonction pour calculer R² ajusté
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Tester tous les sous-ensembles possibles de caractéristiques
for k in range(1, 5):
    for combo in combinations(features, k):
        X = data[list(combo)]
        X = sm.add_constant(X)
        y = data['Age']
        model = sm.OLS(y, X).fit()
        r2_adj = adjusted_r2(model.rsquared, X.shape[0], X.shape[1] - 1)
        results_list.append({'num_features': k, 'features': combo, 'r2_adj': r2_adj})

# Convertir les résultats en DataFrame
results_df = pd.DataFrame(results_list)

# Trouver le meilleur modèle pour chaque nombre de caractéristiques
best_models = results_df.loc[results_df.groupby('num_features')['r2_adj'].idxmax()]

# Afficher les meilleurs modèles
print(best_models)

# Palette de couleurs
colors = ['#F7B267', '#F79D65', '#F4845F', '#F27059']

# Tracer l'histogramme de R² ajusté en fonction du nombre de caractéristiques
fig = go.Figure()

fig.add_trace(go.Bar(
    x=best_models['num_features'],
    y=best_models['r2_adj'],
    text=best_models['r2_adj'].round(2),
    textposition='outside',
    marker_color=colors[:len(best_models)],
    name='R² ajusté'
))

# Personnaliser le graphique
fig.update_layout(
    title='R² ajusté en fonction du nombre de caractéristiques',
    xaxis=dict(title='Nombre de caractéristiques'),
    yaxis=dict(title='R² ajusté'),
    showlegend=False
)

# Afficher le graphique
fig.show()
