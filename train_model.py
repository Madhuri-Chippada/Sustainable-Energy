import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Define the expected dataset file
DATASET_FILE = 'powerconsumption.csv'

# Check for dataset in multiple possible locations
possible_paths = [
    os.path.join(os.getcwd(), DATASET_FILE),  # Current working directory
    os.path.join(os.path.dirname(__file__), DATASET_FILE),  # Script's directory
    os.path.join(os.getcwd(), 'data', DATASET_FILE),  # Optional 'data' subdirectory
]

# Find the dataset file
dataset_path = None
for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        break

if dataset_path is None:
    raise FileNotFoundError(
        f"{DATASET_FILE} not found in current directory ({os.getcwd()}), "
        f"script directory ({os.path.dirname(__file__)}), or 'data' subdirectory. "
        "Please ensure the file is in one of these locations."
    )

print(f"Loading dataset from: {dataset_path}")

# Load the dataset
df = pd.read_csv(dataset_path)

# Define independent and dependent variables
independent_vars = ['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']
dependent_vars = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']

# Process Datetime to extract numerical features
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month

# Handle missing values in Datetime-derived columns
if df[['Hour', 'DayOfWeek', 'Month']].isna().sum().sum() > 0:
    print("Warning: Missing values in Datetime-derived columns. Filling with 0.")
    df[['Hour', 'DayOfWeek', 'Month']] = df[['Hour', 'DayOfWeek', 'Month']].fillna(0)

# Add Datetime-derived features to dependent variables
dependent_vars.extend(['Hour', 'DayOfWeek', 'Month'])

# Create X and y
X = df[independent_vars]
y = df[dependent_vars]

# Handle missing values in X and y
if X.isna().sum().sum() > 0 or y.isna().sum().sum() > 0:
    print("Warning: Missing values detected in X or y. Dropping rows with missing values.")
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[independent_vars]
    y = combined[dependent_vars]

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Save the model
model_path = 'powerconsumption_model.pkl'
joblib.dump(model, model_path)

# Save test data for scatter plot
y_pred = model.predict(x_test)
test_data = {
    'actual_temp': y_test['Temperature'].values.tolist(),
    'predicted_temp': y_pred[:, dependent_vars.index('Temperature')].tolist()
}
test_data_path = 'test_data.pkl'
joblib.dump(test_data, test_data_path)

print("Model and test data saved successfully.")
print(f"Model saved at: {os.path.abspath(model_path)}")
print(f"Test data saved at: {os.path.abspath(test_data_path)}")