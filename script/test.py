import pandas as pd
import matplotlib.pyplot as plt

# Chargez vos données
data_path = 'abalone_data.csv'  # Assurez-vous que ce chemin est correct
abalone_data = pd.read_csv(data_path)

# Calcul de l'âge des ormeaux en ajoutant 1.5 aux anneaux
abalone_data['Age'] = abalone_data['Rings'] + 1.5

# Création d'un histogramme pour visualiser la distribution de l'âge
plt.figure(figsize=(10, 6))
plt.hist(abalone_data['Age'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Abalone Age')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

scatter = px.scatter(abalone_data, x='Length', y='Diameter', color='Sex',
                     labels={'Length': 'Longueur (cm)', 'Diameter': 'Diamètre (cm)'},
                     color_discrete_map={'M': 'red', 'F': 'orange', 'I': 'yellow'})
for trace in scatter['data']:
    fig.add_trace(trace, row=row, col=1)
