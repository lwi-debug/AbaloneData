import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Charger les données
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Calcul de l'âge à partir des anneaux

def remove_outliers(df, column_list):
    for column in column_list:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df

# Appliquer la fonction de suppression des outliers
numeric_cols = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
data = remove_outliers(data, numeric_cols)

# Standardisation des données
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Définition des prédicteurs et de la variable cible
X = data[numeric_cols]  # Exclure 'Rings' et 'Sex'
y = data['Age']

# Ajout de la constante pour l'intercept
X = sm.add_constant(X)

# Création du modèle de régression linéaire multiple
model = sm.OLS(y, X)

# Ajustement du modèle
results = model.fit()

# Affichage du résumé des résultats du modèle
print(results.summary())

# Créer le graphique des coefficients
coefficients = results.params[1:]  # Exclure la constante
features = numeric_cols

fig = go.Figure()

fig.add_trace(go.Bar(
    x=features,
    y=coefficients,
    text=[f"{coef:.4f}" for coef in coefficients],
    textposition='outside',
    marker_color=['#F7B267', '#F79D65', '#F4845F', '#F27059', '#F7B267', '#F79D65', '#F4845F'],
    name='Coefficients'
))

# Personnaliser le graphique
fig.update_layout(
    title='Coefficients de la régression linéaire multiple',
    xaxis=dict(title='Caractéristiques'),
    yaxis=dict(title='Valeur des coefficients'),
    showlegend=False
)

# Afficher le graphique
fig.show()
