import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load your dataset from CSV
file_path = 'rainfall.csv'  # Replace with the actual path
df = pd.read_csv(file_path)  # Assuming your CSV has headers

# Convert values to float
data = df.values.astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Set the number of time steps
n_steps = 3  # You can adjust this value based on your dataset

# Create sequences
X, y = create_sequences(data_scaled, n_steps)

# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# ...

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=100, activation='relu', input_shape=(n_steps, X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(X.shape[2]))

model.compile(optimizer='adam', loss='mse')

# Initialize last_data and prediction_input
last_data = X[-1]

# Manually set the number of epochs
history = model.fit(X, y, epochs=1000, verbose=1, validation_split=0.2)
print(history.history)

# Prediction Loop as a function
def predict_next_year(model, last_data, n_steps, scaler):
    prediction_input = last_data.reshape((1, n_steps, last_data.shape[1]))
    predicted_scaled = model.predict(prediction_input)
    predicted_rainfall = scaler.inverse_transform(predicted_scaled)
    return predicted_rainfall.flatten(), np.vstack([last_data[1:, :], predicted_scaled])

# Predict for the next few years
predicted_values = []

for year in range(2023, 2026):
    predicted_rainfall, last_data = predict_next_year(model, last_data, n_steps, scaler)
    predicted_values.append(predicted_rainfall)

# Convert the list of predicted values to a NumPy array
predicted_values = np.array(predicted_values)

# Plot the predicted rainfall for each month
months = np.arange(1, len(predicted_values[0]) + 1)
years = np.arange(2023, 2026)

plt.figure(figsize=(10, 6))

for i, year in enumerate(years):
    plt.plot(months, predicted_values[i], label=f'Year {year}')

plt.title('Predicted Rainfall (2023-2025)')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()
