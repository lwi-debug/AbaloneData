import pandas as pd

# Chargement des données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)

# Compter le nombre d'observations dans chaque catégorie de la variable 'Sex'
sex_counts = data['Sex'].value_counts()

# Affichage des résultats
print(sex_counts)
