import pandas as pd
import plotly.express as px

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

# Créer un DataFrame pour les corrélations
age_corr_df = pd.DataFrame(age_correlations).reset_index()
age_corr_df.columns = ['Variable', 'Correlation']

# Palette de couleurs
colors = ['#355070', '#515575', '#6D597A', '#915F78', '#B56576', '#E56B6F', '#E88C7D', '#EAAC8B']

# Créer le graphique
fig = px.bar(age_corr_df, x='Variable', y='Correlation',
             title='Coefficients de corrélation des variables par rapport à l\'âge',
             color='Correlation',
             color_continuous_scale=colors,
             labels={'Correlation': 'Coefficient de Corrélation', 'Variable': 'Variable'})

# Personnaliser l'apparence
fig.update_layout(title={'x':0.5},
                  xaxis_title="Variables",
                  yaxis_title="Coefficient de Corrélation",
                  coloraxis_showscale=False)

# Afficher le graphique
fig.show()
