import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données
data_path = "abalone_data.csv"  # Remplacez par le chemin réel de votre fichier de données
abalone_data = pd.read_csv(data_path)

# Calcul des statistiques descriptives
descriptive_stats = abalone_data.describe().round(2)
print("Statistiques Descriptives:")
print(descriptive_stats)

# Identification des variables qualitatives et de leurs catégories
qualitative_vars = abalone_data.select_dtypes(include=['object']).columns
print("Variables qualitatives :", qualitative_vars)

for var in qualitative_vars:
    print(f"Catégories dans {var} :", abalone_data[var].unique())

# Histogrammes pour toutes les variables quantitatives
abalone_data.hist(bins=15, figsize=(15, 10), layout=(3, 3))
plt.suptitle('Histogrammes des Variables Quantitatives')
plt.show()

# Boîtes à moustaches pour les variables quantitatives
quantitative_vars = abalone_data.select_dtypes(include=['float64', 'int64']).columns
for var in quantitative_vars:
    plt.figure(figsize=(10, 5))
    plt.boxplot(abalone_data[var])
    plt.title(f'Boîte à moustaches de {var}')
    plt.xlabel(var)
    plt.ylabel('Valeurs (unité varie)')
    plt.grid(True)
    plt.show()

# Diagramme de dispersion entre la Longueur et le Diamètre
plt.figure(figsize=(10, 6))
plt.scatter(abalone_data['Length'], abalone_data['Diameter'])
plt.title('Diagramme de Dispersion entre la Longueur et le Diamètre')
plt.xlabel('Longueur (cm)')
plt.ylabel('Diamètre (cm)')
plt.grid(True)
plt.show()
# -*- coding: utf-8 -*-

