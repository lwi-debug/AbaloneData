import pandas as pd

# Load the data
data_path = "abalone_data.csv"  # Replace this with your actual data file path
abalone_data = pd.read_csv(data_path)

# Check for missing values in the dataset
missing_values = abalone_data.isnull().sum()

# Print the number of missing values per variable
print("Number of missing values per variable:")
print(missing_values)

# Check if there are any missing values in the dataset
if abalone_data.isnull().values.any():
    print("There are missing values in the dataset.")
    # Delete observations with missing values
    abalone_data.dropna(inplace=True)
    print("Observations with missing values have been deleted.")
else:
    print("There are no missing values in the dataset.")

# Optionally, save the cleaned data back to a CSV
abalone_data.to_csv("path_to_your_cleaned_data.csv", index=False)  # Update the path as needed
