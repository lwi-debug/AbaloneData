import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import t
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the data
data_path = 'abalone_data.csv'  # Ensure this path points to your data file
abalone_data = pd.read_csv(data_path)

# Add the 'Age' variable
abalone_data['Age'] = abalone_data['Rings'] + 1.5

# Identify and remove outliers based on IQR in 'Shell weight'
Q1 = abalone_data['Shell weight'].quantile(0.25)
Q3 = abalone_data['Shell weight'].quantile(0.75)
IQR = Q3 - Q1
filter = (abalone_data['Shell weight'] >= Q1 - 1.5 * IQR) & (abalone_data['Shell weight'] <= Q3 + 1.5 * IQR)
abalone_data_filtered = abalone_data[filter]

# Prepare data for regression
X = abalone_data_filtered[['Shell weight']]  # Independent variable
Y = abalone_data_filtered['Age']             # Dependent variable, target

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Compute the standard error of the coefficients using statsmodels for detailed summary
X_with_intercept = sm.add_constant(X_train)  # Adding intercept term for calculations
model_sm = sm.OLS(Y_train, X_with_intercept).fit()  # Refit with statsmodels to get SE easily
summary = model_sm.summary()
print(summary)

# Extract the standard error directly from the summary using .iloc for future-proofing
standard_error_beta1 = model_sm.bse.iloc[1]  # Index 1 refers to the first coefficient after the intercept

# Calculate the t critical value for 95% CI
t_critical = t.ppf(0.975, df=len(X_train) - 2)  # 0.975 for upper tail and n-2 degrees of freedom

# Calculate the confidence interval for beta1
confidence_interval = model.coef_[0] + np.array([-1, 1]) * t_critical * standard_error_beta1

print("95% Confidence Interval for Beta1:", confidence_interval)

# Print model's performance metrics based on the test set
mse = np.mean((predictions - Y_test) ** 2)
r2 = model.score(X_test, Y_test)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
