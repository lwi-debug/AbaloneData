import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Chargement des données
data_path = "abalone_data.csv"
abalone_data = pd.read_csv(data_path)

# Variables quantitatives
quantitative_vars = abalone_data.select_dtypes(include=['float64', 'int64']).columns

# Palette de couleurs personnalisée
colors = {
    'Length': '#F94144',  # Red
    'Diameter': '#F3722C',  # Light Red
    'Height': '#F8961E',  # Lighter Red
    'Whole weight': '#F9844A',  # Lightest Red
    'Shucked weight': '#F9C74F',  # Near White Red
    'Viscera weight': '#90BE6D',  # Dark Green
    'Shell weight': '#43AA8B',  # Darker Green
    'Rings': '#4D908E'  # Darkest Green
}

# Création du subplot
fig = make_subplots(rows=len(quantitative_vars) + 1, cols=2,
                    specs=[[{"type": "histogram"}, {"type": "box"}] for _ in quantitative_vars] + [[{"colspan": 2}, None]],
                    horizontal_spacing=0.05, vertical_spacing=0.05)

# Ajout de l'histogramme et du box plot pour chaque variable quantitative
row = 1
for var in quantitative_vars:
    # Histogramme
    fig.add_trace(go.Histogram(x=abalone_data[var], nbinsx=30, name=f'{var} Histogram', marker_color=colors[var]), row=row, col=1)
    fig.update_xaxes(title_text=var, row=row, col=1)  # Ajout du titre sous l'histogramme

    # Box plot
    fig.add_trace(go.Box(y=abalone_data[var], name=f'{var} Box Plot', boxpoints='all', jitter=0.5, pointpos=-1.8, marker_color=colors[var]), row=row, col=2)
    fig.update_yaxes(title_text=var, row=row, col=2)  # Ajout du titre sous le box plot

    row += 1

# Ajout du diagramme de dispersion entre la Longueur et le Diamètre
scatter = px.scatter(abalone_data, x='Length', y='Diameter', color='Sex',
                     labels={'Length': 'Longueur (cm)', 'Diameter': 'Diamètre (cm)'},
                     color_discrete_map={'M': '#F94144', 'F': '#F8961E', 'I': '#43AA8B'})
for trace in scatter['data']:
    fig.update_xaxes(title_text="Longueur vs Diamètre", row=row, col=1)
    fig.add_trace(trace, row=row, col=1)


# Mise à jour du layout pour améliorer la présentation
fig.update_layout(height=3000, showlegend=True, title_text="Abalone Data Analysis: Descriptive Statistics and Relationships")

# Affichage du graphique
fig.show()
