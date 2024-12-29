# Predicting Repair Time and Next Service Date for Refrigeration Units

This project aims to predict the repair time (TAT) for refrigeration units and suggest the next service date based on the predicted repair time. It utilizes an LSTM (Long Short-Term Memory) neural network to make accurate predictions based on historical repair data.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Prediction](#prediction)
- [Output](#output)

## Features
- Predicts repair time (TAT) for refrigeration units.
- Suggests the next service date based on the predicted repair time.
- Includes relevant information such as device ID, date, original repair time, predicted repair time, next service date, and failure type in the output.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Openpyxl

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Place your data files in the specified paths.
4. Run the script to train the model and make predictions.

## Data Preparation
The input data should be in CSV format with the following columns:
- `Date`: The date of the service.
- `Device_ID`: The identifier for each refrigeration unit.
- `Repair_Time`: The time taken to repair the unit.
- `Service_Cost`: The cost of the service.
- `Failure_Type`: The type of failure that occurred.

### Example Data
```csv
Date,Device_ID,Repair_Time,Service_Cost,Failure_Type
2024-01-01,Device_1,11,1684,Compressor Issue
2024-01-02,Device_18,2,3404,Power Supply Failure
...
