import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import time

# Load data
file_path = 'C:/Users/User/Desktop/nahom/Maths/multipliers.csv'
df = pd.read_csv(file_path, header=None, names=['Multiplier'], dtype={'Multiplier': str}, low_memory=False)
df['Multiplier'] = pd.to_numeric(df['Multiplier'], errors='coerce')
df = df.dropna().reset_index(drop=True)

# Advanced Feature Engineering with lag of 10 and rolling statistics
def create_features(data, lag=10):
    X, y = [], []
    for i in range(len(data) - lag):
        lags = data[i:i + lag]
        rolling_mean = np.mean(lags)
        rolling_std = np.std(lags)
        features = list(lags) + [rolling_mean, rolling_std]
        X.append(features)
        y.append(data[i + lag])
    return np.array(X), np.array(y)

# Prepare the data
lag = 10
X, y = create_features(df['Multiplier'].values, lag)
X = X.reshape(X.shape[0], -1)  # Flatten the input for SVR

# Normalize the features and target variable
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Expanded Hyperparameter Tuning with GridSearchCV
param_grid = {
    'svr__C': [0.1, 1, 10, 100, 1000, 10000],
    'svr__epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
    'svr__kernel': ['linear', 'rbf', 'poly'],
    'svr__gamma': ['scale', 'auto']  # Add gamma parameter for more control in non-linear models
}
model = make_pipeline(StandardScaler(), SVR())
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# Save the best model
joblib.dump(best_model, 'svr_model_v2.pkl')

# Predict the next multiplier
def predict_next_multiplier(recent_multipliers):
    recent_multipliers = np.array(recent_multipliers).reshape(1, -1)
    recent_multipliers = scaler_X.transform(recent_multipliers)
    next_multiplier_scaled = best_model.predict(recent_multipliers)[0]
    next_multiplier = scaler_y.inverse_transform([[next_multiplier_scaled]])[0, 0]
    return float(max(next_multiplier, 1.00))

# Function to validate and clean input
def get_valid_multiplier(prompt):
    while True:
        try:
            multiplier = input(prompt)
            multiplier = ''.join(filter(str.isdigit, multiplier.replace('.', '', 1)))
            return float(multiplier) if multiplier else 1.00
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Prompt the user for the last 10 multipliers
recent_multipliers = []
for i in range(1, 11):
    multiplier = get_valid_multiplier(f'Multiplier {i}: ')
    recent_multipliers.append(multiplier)

# Predict the next multiplier
start_time = time.time()
predicted_multiplier = predict_next_multiplier(recent_multipliers)
end_time = time.time()

# Display the prediction
print(f'Predicted next multiplier: {predicted_multiplier:.2f}')
print(f'Prediction took {end_time - start_time:.2f} seconds')

# Continuously ask for the most recent multiplier and predict the next one
while True:
    recent_multiplier = get_valid_multiplier('Most recent multiplier: ')
    recent_multipliers.pop(0)
    recent_multipliers.append(recent_multiplier)
    
    start_time = time.time()
    predicted_multiplier = predict_next_multiplier(recent_multipliers)
    end_time = time.time()
    
    print(f'Predicted next multiplier: {predicted_multiplier:.2f}')
    print(f'Prediction took {end_time - start_time:.2f} seconds')
