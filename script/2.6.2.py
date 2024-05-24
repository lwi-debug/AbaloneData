import pandas as pd
import plotly.express as px

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)

# Ajouter la variable 'Age'
data['Age'] = data['Rings'] + 1.5

# Créer un boxplot de la variable cible 'Age' en fonction de la variable 'Sex'
fig = px.box(data, x='Sex', y='Age', color='Sex',
             title='Boxplot de l\'âge des ormeaux par sexe',
             color_discrete_map={'M': '#F94144', 'F': '#F8961E', 'I': '#43AA8B'})

# Afficher le graphique
fig.show()
