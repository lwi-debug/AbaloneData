import pandas as pd

# Chargement des données
data_path = 'abalone_data.csv'  # Assurez-vous que le chemin vers votre fichier de données est correct
abalone_data = pd.read_csv(data_path)

# Ajout de la variable 'Age'
abalone_data['Age'] = abalone_data['Rings'] + 1.5

# Suppression des colonnes 'Sex' et 'Rings' pour le calcul de corrélation
abalone_data_numeric = abalone_data.drop(columns=['Sex', 'Rings'])

# Calcul des coefficients de corrélation
correlation_matrix = abalone_data_numeric.corr()

# Affichage des corrélations de 'Age' avec les autres variables
age_correlations = correlation_matrix['Age'].sort_values(ascending=False)
print(age_correlations)
# -*- coding: utf-8 -*-

