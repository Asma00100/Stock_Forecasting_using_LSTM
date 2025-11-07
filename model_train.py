import pandas as pd
import pymysql
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import joblib
from dotenv import load_dotenv
import os

print("LSTM Model Training Script Started")

# =============================
# LOAD ENVIRONMENT VARIABLES
# =============================
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))

# ---------- Step 1: Connect to MySQL ----------
try:
    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    print("Connected to MySQL successfully!")
except Exception as e:
    print("Error connecting to MySQL:", e)
    exit()

# ---------- Step 2: Fetch data ----------
try:
    query = "SELECT date, close FROM aapl_stock ORDER BY date ASC"
    df = pd.read_sql(query, connection, parse_dates=['date'])
    print(f"Data fetched from MySQL ({len(df)} rows)")
except Exception as e:
    print("Error fetching data:", e)
    exit()
finally:
    connection.close()

# ---------- Step 3: Prepare data ----------
df = df.dropna()
close_prices = df['close'].values.reshape(-1, 1)

# Normalize the closing prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create time sequences (60 past values → 1 future value)
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

print("Data ready for training")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------- Step 4: Build LSTM model ----------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("Model compiled successfully")

# ---------- Step 5: Train the model ----------
history = model.fit(
    X, y,
    epochs=10,           # increase to 30-50 later if needed
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ---------- Step 6: Predictions for visualization ----------
predicted_scaled = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_scaled)
real_prices = df['close'].values[sequence_length:]

# ---------- Step 7: Plot results ----------
plt.figure(figsize=(12, 5))
plt.plot(real_prices, color='blue', label='Actual AAPL Price')
plt.plot(predicted_prices, color='red', label='Predicted AAPL Price')
plt.title('AAPL Stock Price Prediction (5-min Interval, Alpha Vantage Data)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Step 8: Save model & scaler ----------
model.save("aapl_lstm_model.h5")
joblib.dump(scaler, "aapl_scaler.save")

print("Model saved as aapl_lstm_model.h5")
print("Scaler saved as aapl_scaler.save")
print("Training completed successfully!")
