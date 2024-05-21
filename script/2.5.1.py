import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load your data
data_path = 'abalone_data.csv'
data = pd.read_csv(data_path)
data['Age'] = data['Rings'] + 1.5  # Assuming Age calculation is required

# Define predictors and the response variable
X = data.drop(columns=['Rings', 'Age', 'Sex'])  # Exclude non-numeric or target variables
y = data['Age']

# Function to fit model and get R squared
def fit_model(X, y):
    X = sm.add_constant(X)  # adding a constant
    model = sm.OLS(y, X).fit()
    return model.rsquared_adj

# Store results
results = []

# Generate all combinations of 1 to 4 features
for k in range(1, 5):
    for combo in itertools.combinations(X.columns, k):
        combo = list(combo)
        X_subset = X[combo]
        adj_r_squared = fit_model(X_subset, y)
        results.append((combo, adj_r_squared))

# Sort results by best adjusted R-squared
results.sort(key=lambda x: x[1], reverse=True)

# Plotting the best R squared for each number of features
plt.figure(figsize=(10, 6))
for k in range(1, 5):
    best_adj_r_squared = max([r[1] for r in results if len(r[0]) == k])
    plt.bar(k, best_adj_r_squared)

plt.xlabel('Number of Features')
plt.ylabel('Adjusted R²')
plt.title('Best Adjusted R² for Each Number of Features')
plt.xticks([1, 2, 3, 4])
plt.show()

# Output the best overall model
best_overall = results[0]
print("Best model:", best_overall[0])
print("Adjusted R² of best model:", best_overall[1])
