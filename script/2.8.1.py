import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Simuler des données temporelles
np.random.seed(0)
date_rng = pd.date_range(start='1/1/2000', end='1/1/2021', freq='M')
length_series = np.random.normal(loc=0.5, scale=0.1, size=len(date_rng))
diameter_series = np.random.normal(loc=0.3, scale=0.05, size=len(date_rng))

# Ajouter une tendance croissante
length_series = np.cumsum(length_series)
diameter_series = np.cumsum(diameter_series)

# Créer un DataFrame
time_series_data = pd.DataFrame(date_rng, columns=['date'])
time_series_data['length'] = length_series
time_series_data['diameter'] = diameter_series

# Ajuster le modèle ARIMA pour la longueur
model_length = ARIMA(time_series_data['length'], order=(5, 1, 0))
model_length_fit = model_length.fit()

# Ajuster le modèle ARIMA pour le diamètre
model_diameter = ARIMA(time_series_data['diameter'], order=(5, 1, 0))
model_diameter_fit = model_diameter.fit()

# Faire des prévisions
forecast_steps = 24  # Prévoir les 2 prochaines années (24 mois)
forecast_length = model_length_fit.forecast(steps=forecast_steps)
forecast_diameter = model_diameter_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=time_series_data['date'].iloc[-1] + timedelta(days=30), periods=forecast_steps, freq='M')

# Ajouter les prévisions au DataFrame
forecast_df = pd.DataFrame(forecast_index, columns=['date'])
forecast_df['forecast_length'] = forecast_length.values
forecast_df['forecast_diameter'] = forecast_diameter.values

# Visualiser les données réelles et prévues avec Plotly
fig = go.Figure()

# Tracer les données réelles
fig.add_trace(go.Scatter3d(
    x=time_series_data['date'],
    y=time_series_data['length'],
    z=time_series_data['diameter'],
    mode='markers',
    name='Données Réelles',
    marker=dict(size=4, color='#FFBF00')
))

# Tracer les prévisions
fig.add_trace(go.Scatter3d(
    x=forecast_df['date'],
    y=forecast_df['forecast_length'],
    z=forecast_df['forecast_diameter'],
    mode='markers',
    name='Prévisions ARIMA',
    marker=dict(size=4, color='#E83F6F')
))

# Personnaliser le graphique
fig.update_layout(
    title='Prévisions des Tendances de Longueur et Diamètre des Ormeaux avec ARIMA',
    scene=dict(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Longueur'),
        zaxis=dict(title='Diamètre')
    ),
    showlegend=True
)

# Afficher le graphique
fig.show()
 