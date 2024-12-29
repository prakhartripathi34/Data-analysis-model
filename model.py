import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the CSV file
file_path = 'refrigeration_service_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Feature engineering: Extract relevant features
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Create lag features
data['Lag_1'] = data['Repair_Time'].shift(1)
data['Lag_2'] = data['Repair_Time'].shift(2)

# Create rolling averages
data['Rolling_Mean_3'] = data['Repair_Time'].rolling(window=3).mean()
data['Rolling_Mean_6'] = data['Repair_Time'].rolling(window=6).mean()

# Handle missing values
data['Service_Cost'] = data['Service_Cost'].fillna(data['Service_Cost'].mean())
data['Repair_Time'] = data['Repair_Time'].fillna(data['Repair_Time'].mean())
data['Lag_1'] = data['Lag_1'].fillna(data['Lag_1'].mean())
data['Lag_2'] = data['Lag_2'].fillna(data['Lag_2'].mean())
data['Rolling_Mean_3'] = data['Rolling_Mean_3'].fillna(data['Rolling_Mean_3'].mean())
data['Rolling_Mean_6'] = data['Rolling_Mean_6'].fillna(data['Rolling_Mean_6'].mean())

# Ensure correct data types for numerical and categorical columns
data['Service_Cost'] = data['Service_Cost'].astype(float)
data['Repair_Time'] = data['Repair_Time'].astype(float)
data['Failure_Type'] = data['Failure_Type'].astype(str)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['Failure_Type'], drop_first=True)

# Select features and target variable
features = ['Month', 'Year', 'Service_Cost', 'Lag_1', 'Lag_2', 'Rolling_Mean_3', 'Rolling_Mean_6'] + [col for col in data.columns if col.startswith('Failure_Type')]
target = 'Repair_Time'  # Assuming Repair_Time is the target variable for Service TAT

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Initialize the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=1)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Example new data for prediction
new_data = pd.DataFrame({
    'Month': [1],
    'Year': [2025],
    'Service_Cost': [1500],
    'Lag_1': [5.5],  # Example lag value
    'Lag_2': [6.0],  # Example lag value
    'Rolling_Mean_3': [6.3],  # Example rolling mean value
    'Rolling_Mean_6': [6.2],  # Example rolling mean value
    'Failure_Type_Compressor Issue': [0],
    'Failure_Type_Coolant Leak': [0],
    'Failure_Type_Other': [0],
    'Failure_Type_Power Supply Failure': [0],
    'Failure_Type_Sensor Fault': [1]
})

# Ensure new_data has the same structure as the training data
for col in X_train.columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Reorder columns to match the training data
new_data = new_data[X_train.columns]

# Standardize the new data
new_data_scaled = scaler.transform(new_data)
new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

# Make predictions
predicted_tat = model.predict(new_data_scaled)
print(f'Predicted Service TAT: {predicted_tat[0][0]} hours')
