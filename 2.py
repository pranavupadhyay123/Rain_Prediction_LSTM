import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Named Constants
N_STEPS = 3
EPOCHS = 100

def load_data(file_path):
    df = pd.read_csv(file_path, header=None).iloc[1:]
    data = df.values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def build_lstm_model(n_steps, input_shape):
    model = Sequential()
    model.add(LSTM(units=200, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=150, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(input_shape[1]))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(data.shape[0]):
        end_ix = i + n_steps
        if end_ix > data.shape[0] - 1:
            break
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def main():
    file_path = 'rainfall.csv'  # Replace with the actual path
    data_scaled, scaler = load_data(file_path)

    X, y = create_sequences(data_scaled, N_STEPS)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    model = build_lstm_model(N_STEPS, (N_STEPS, X.shape[2]))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    predictions = []
    last_data = data_scaled[-N_STEPS:, :]
    for year in range(2023, 2026):
        for month_step in range(1, 13):
            prediction_input = last_data.reshape((1, N_STEPS, X.shape[2]))
            predicted_scaled = model.predict(prediction_input)
            predicted_rainfall = scaler.inverse_transform(predicted_scaled)
            predictions.append(predicted_rainfall.flatten())
            last_data = np.vstack([last_data[1:, :], predicted_scaled])

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_ticks = [f"{year}-{month}" for year in range(2023, 2026) for month in months]

    plt.figure(figsize=(15, 6))
    for i, col in enumerate(pd.read_csv(file_path, header=None).iloc[0].index):
        plt.plot(x_ticks, [last_data[-1, i]] + [pred[i] for pred in predictions[:-1]], label=f"Station {i + 1}")

    plt.title('Predicted Rainfall for 2023, 2024, and 2025')
    plt.xlabel('Month')
    plt.ylabel('Rainfall')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
