import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Calcul de l'âge à partir des anneaux

def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Retirer les outliers de la variable 'Shell weight'
data = remove_outliers(data, 'Shell weight')

# Standardisation des données
scaler = StandardScaler()
data[['Shell weight', 'Age']] = scaler.fit_transform(data[['Shell weight', 'Age']])

# Sélection des variables
X = data['Shell weight']  # Variable explicative
Y = data['Age']           # Variable dépendante

# Ajout de la constante pour l'intercept
X = sm.add_constant(X)

# Création du modèle de régression linéaire
model = sm.OLS(Y, X)

# Ajustement du modèle
results = model.fit()

# Affichage du résumé des résultats du modèle
print(results.summary())

# Définir la palette de couleurs pour le sexe
color_map = {'M': '#F94144', 'F': '#F8961E', 'I': '#ffcb69'}

# Ajouter une colonne de couleur au DataFrame
data['color'] = data['Sex'].map(color_map)

# Graphique de régression linéaire avec Plotly
fig = px.scatter(data, x='Shell weight', y='Age', color='Sex', color_discrete_map=color_map, title='Régression Linéaire: Âge vs. Poids de la Coquille')

# Ajouter la ligne de régression
fig.add_trace(go.Scatter(x=data['Shell weight'], y=results.fittedvalues, mode='lines', name='Ligne de Régression', line=dict(color='#43AA8B')))

# Afficher le graphique
fig.show()
