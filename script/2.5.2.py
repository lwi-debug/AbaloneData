import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Chargement des données
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

# Définition des prédicteurs et de la variable cible après exclusion des outliers
X = data.drop(columns=['Rings', 'Age', 'Sex'])  # Exclure les variables non numériques ou cibles
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
plt.figure(figsize=(10, 6))
for k in range(1, 5):
    best_adj_r_squared = max([r[1] for r in results if len(r[0]) == k])
    plt.bar(k, best_adj_r_squared)

plt.xlabel('Nombre de caractéristiques')
plt.ylabel('R² Ajusté')
plt.title('Meilleur R² Ajusté pour Chaque Nombre de Caractéristiques')
plt.xticks([1, 2, 3, 4])
plt.show()

# Affichage du meilleur modèle global
best_overall = results[0]
print("Meilleur modèle:", best_overall[0])
print("R² ajusté du meilleur modèle:", best_overall[1])

# Affichage des résultats du modèle
print(best_overall[2].summary())
