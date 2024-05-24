import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# Créer le graphique avec les couleurs de la palette
fig = go.Figure()

for i, row in age_corr_df.iterrows():
    fig.add_trace(go.Bar(
        x=[row['Variable']],
        y=[row['Correlation']],
        name=row['Variable'],
        text=f"{row['Correlation']:.2f}",
        textposition='outside',
        marker_color=colors[i % len(colors)]  # Assigner une couleur de la palette
    ))

# Personnaliser l'apparence
fig.update_layout(
    title='Coefficients de corrélation des variables par rapport à l\'âge',
    title_x=0.5,
    xaxis_title="Variables",
    yaxis_title="Coefficient de Corrélation",
    showlegend=True
)

# Afficher le graphique
fig.show()
