import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Calcul de l'âge

# Identification et exclusion des outliers pour toutes les variables quantitatives
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

# Définition des prédicteurs et de la variable cible après exclusion des outliers
X = data[numeric_cols]
y = data['Age']

# Fonction pour ajuster le modèle et obtenir le R² ajusté
def fit_model(X, y):
    X = sm.add_constant(X)  # ajout d'une constante
    model = sm.OLS(y, X).fit()
    return model.rsquared_adj, model

# Stockage des résultats
results = []

# Génération de toutes les combinaisons de 1 à 4 caractéristiques
for k in range(1, 5):
    for combo in itertools.combinations(X.columns, k):
        combo = list(combo)
        X_subset = X[combo]
        adj_r_squared, model = fit_model(X_subset, y)
        results.append((combo, adj_r_squared, model))

# Tri des résultats par le meilleur R² ajusté
results.sort(key=lambda x: x[1], reverse=True)

# Affichage du meilleur R² pour chaque nombre de caractéristiques
best_combination = results[0][0]
best_model = results[0][2]

# Afficher les résultats
print("Meilleur modèle avec les caractéristiques:", best_combination)
print("R² ajusté du meilleur modèle:", results[0][1])
print(best_model.summary())

# Créer le graphique des coefficients
coefficients = best_model.params[1:]  # Exclure la constante
features = best_combination

fig = go.Figure()

fig.add_trace(go.Bar(
    x=features,
    y=coefficients,
    text=[f"{coef:.4f}" for coef in coefficients],
    textposition='outside',
    marker_color=['#F7B267', '#F79D65', '#F4845F', '#F27059'],
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
