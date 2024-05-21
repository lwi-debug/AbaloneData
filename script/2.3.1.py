import pandas as pd

# Chargement des données
data_path = "abalone_data.csv"  # Remplacez par le chemin réel de votre fichier de données
abalone_data = pd.read_csv(data_path)

# Suppression de la variable 'Sex' car nous nous concentrons uniquement sur les variables quantitatives
abalone_data_numeric = abalone_data.drop(columns=['Sex'])

# Calcul de la variance pour chaque variable
variances = abalone_data_numeric.var()
print("Variances des variables:")
print(variances)

# Décision sur la standardisation
print("\nÉtant donné les variances ci-dessus, est-il nécessaire de standardiser les variables ?")
if variances.max() / variances.min() > 10:
    print("Oui, il est recommandé de standardiser les variables car les écarts de variance sont grands.")
else:
    print("Non, la standardisation peut ne pas être nécessaire si les variances sont relativement similaires.")
