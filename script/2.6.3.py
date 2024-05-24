import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Calcul de l'âge

# Inclure toutes les variables quantitatives nécessaires
numeric_cols = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
data = data[numeric_cols + ['Age']]

# Standardisation des données
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Définition des prédicteurs et de la variable cible
X = data[numeric_cols]
y = data['Age']

# Ajustement du modèle de régression
X = sm.add_constant(X)  # ajout d'une constante
model = sm.OLS(y, X).fit()

# Préparer les nouvelles données pour la prédiction
new_abalone = pd.DataFrame({
    'Length': [0.35],
    'Diameter': [0.265],
    'Height': [0.09],
    'Whole weight': [0.2255],  # Vous devez fournir cette valeur
    'Shucked weight': [0.0995],  # Vous devez fournir cette valeur
    'Viscera weight': [0.0485],  # Vous devez fournir cette valeur
    'Shell weight': [0.07]
})

# Standardisation des nouvelles données
new_abalone[numeric_cols] = scaler.transform(new_abalone[numeric_cols])

# Ajout de la constante à la nouvelle observation
new_abalone = sm.add_constant(new_abalone, has_constant='add')

# Effectuer la prédiction
predicted_age = model.predict(new_abalone)
print("Âge prédit pour l'abalone infant:", predicted_age[0])
