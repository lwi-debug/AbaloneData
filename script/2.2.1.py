# Let's load the data from the uploaded CSV file to check the number of observations and variables.
import pandas as pd

# Load the data
data_path = "abalone_data.csv"
abalone_data = pd.read_csv(data_path)

# Check the first few rows and the structure of the data
abalone_data.head(), abalone_data.shape
