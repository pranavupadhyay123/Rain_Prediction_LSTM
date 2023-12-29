import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load your dataset from CSV
file_path = 'rainfall.csv'  # Replace with the actual path
df = pd.read_csv(file_path, header=None)

# Skip the header row
df = df.iloc[1:]

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

# Build a more complex LSTM model
model = Sequential()
model.add(LSTM(units=200, activation='relu', input_shape=(n_steps, X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))  # Adding dropout for regularization
model.add(LSTM(units=150, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(X.shape[2]))

model.compile(optimizer='adam', loss='mse')

# Fit the model to the data for a longer duration
model.fit(X, y, epochs=100, verbose=1)

# Predict rainfall for 2023, 2024, and 2025 with finer granularity
predictions = []
last_data = data_scaled[-n_steps:, :]
for year in range(2023, 2026):
    # Predict for each month within the year
    for month_step in range(1, 13):
        prediction_input = last_data.reshape((1, n_steps, X.shape[2]))
        predicted_scaled = model.predict(prediction_input)

        # Inverse transform the predicted data to get the actual rainfall values
        predicted_rainfall = scaler.inverse_transform(predicted_scaled)

        # Append the prediction to the list
        predictions.append(predicted_rainfall.flatten())

        # Update the input sequence for the next prediction
        last_data = np.vstack([last_data[1:, :], predicted_scaled])

# Plot the predicted rainfall values
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x_ticks = [f"{year}-{month}" for year in range(2023, 2026) for month in months]

plt.figure(figsize=(15, 6))
for i in range(len(df.columns)):
    plt.plot(x_ticks, [data[-1, i]] + [pred[i] for pred in predictions[:-1]], label=f"Station {i + 1}")

plt.title('Predicted Rainfall for 2023, 2024, and 2025')
plt.xlabel('Month')
plt.ylabel('Rainfall')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend()
plt.show()
