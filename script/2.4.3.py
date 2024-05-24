import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import t

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

# Calcul de l'intervalle de confiance à 95% pour le coefficient β1
beta1 = results.params[1]
se_beta1 = results.bse[1]
t_critical = t.ppf(1 - 0.025, df=results.df_resid)
ci_lower = beta1 - t_critical * se_beta1
ci_upper = beta1 + t_critical * se_beta1
print(f"95% Confidence Interval for Beta1: [{ci_lower}, {ci_upper}]")

# Test d'hypothèse pour la pente nulle (H0: β1 = 0)
t_stat = results.tvalues[1]
p_value = results.pvalues[1]
print(f"T-statistic for Beta1: {t_stat}")
print(f"P-value for Beta1: {p_value}")

if p_value < 0.05:
    print("Reject the null hypothesis: The coefficient β1 is significantly different from zero.")
else:
    print("Fail to reject the null hypothesis: The coefficient β1 is not significantly different from zero.")

# Coefficient de détermination R²
r_squared = results.rsquared
print(f"Coefficient of Determination (R²): {r_squared}")

# Définir la palette de couleurs pour le sexe
color_map = {'M': '#F94144', 'F': '#F8961E', 'I': '#43AA8B'}

# Ajouter une colonne de couleur au DataFrame
data['color'] = data['Sex'].map(color_map)

# Graphique de régression linéaire avec Plotly
fig = px.scatter(data, x='Shell weight', y='Age', color='Sex', color_discrete_map=color_map, title='Régression Linéaire: Âge vs. Poids de la Coquille')

# Ajouter la ligne de régression
fig.add_trace(go.Scatter(x=data['Shell weight'], y=results.fittedvalues, mode='lines', name='Ligne de Régression', line=dict(color='#7b2cbf')))

# Afficher le graphique
fig.show()
