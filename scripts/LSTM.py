import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def load_data(file_path, date_column, value_column):
    """Load and preprocess time series data."""
    df = pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
    df = df.asfreq('D')  # Adjust frequency as needed
    return df[[value_column]]

def preprocess_data(series, look_back=10):
    """Scale and create sequences for LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series)
    X, y = [], []
    for i in range(len(series_scaled) - look_back):
        X.append(series_scaled[i:i + look_back])
        y.append(series_scaled[i + look_back])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """Build an LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- LSTM Model Building Module (lstm_model.py) ---
def build_lstm_model(input_shape):
    """Builds an LSTM model."""
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32, verbose=0):
    """Trains an LSTM model."""
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# --- Model Evaluation Module (model_evaluation.py) ---
def evaluate_lstm_model(model, X_test, y_test, scaler):
    """Evaluates an LSTM model."""
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = sqrt(mean_squared_error(y_test_original, predictions))
    return rmse, predictions, y_test_original
    
def plot_forecast(series, forecast_values):
    """Plot actual and forecasted values."""
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='Actual Data')
    plt.plot(pd.date_range(start=series.index[-1], periods=len(forecast_values)+1, freq='D')[1:], 
             forecast_values, label='Forecast', color='red')
    plt.legend()
    plt.show()