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

def create_sequences(data, n_steps, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train, test = data[0:train_size, :], data[train_size:len(data), :]
    X_train, y_train = [], []
    X_test, y_test = [], []
    for i in range(len(train)):
        end_ix = i + n_steps
        if end_ix > len(train) - 1:
            break
        seq_x, seq_y = train[i:end_ix, :], train[end_ix, :]
        X_train.append(seq_x)
        y_train.append(seq_y)
    for i in range(len(test)):
        end_ix = i + n_steps
        if end_ix > len(test) - 1:
            break
        seq_x, seq_y = test[i:end_ix, :], test[end_ix, :]
        X_test.append(seq_x)
        y_test.append(seq_y)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def main():
    file_path = 'rainfall.csv'  # Replace with the actual path
    data_scaled, scaler = load_data(file_path)

    X_train, y_train, X_val, y_val = create_sequences(data_scaled, N_STEPS)

    model = build_lstm_model(N_STEPS, (N_STEPS, X_train.shape[2]))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

    predictions = []
    last_data = data_scaled[-N_STEPS:, :]
    for year in range(2023, 2026):
        for month_step in range(1, 13):
            prediction_input = last_data.reshape((1, N_STEPS, X_val.shape[2]))
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
