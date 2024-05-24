import pandas as pd
import plotly.express as px

# Chargement des données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)

# Compter le nombre d'observations dans chaque catégorie de la variable 'Sex'
sex_counts = data['Sex'].value_counts().reset_index()
sex_counts.columns = ['Sex', 'Count']

# Définir la palette de couleurs pour chaque catégorie
color_map = {'M': '#F94144', 'F': '#F8961E', 'I': '#43AA8B'}

# Créer un graphique à barres avec Plotly
fig = px.bar(sex_counts, x='Sex', y='Count', color='Sex', title="Nombre d'observations par catégorie de sexe",
             color_discrete_map=color_map, text='Count')

# Mettre à jour les traces pour afficher les chiffres au-dessus des barres
fig.update_traces(textposition='outside')

# Afficher le graphique
fig.show()
