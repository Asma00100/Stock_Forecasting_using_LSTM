import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import pymysql
import matplotlib.pyplot as plt
from datetime import timedelta
import requests
import os
from dotenv import load_dotenv

# ------------------ LOAD ENV ------------------
load_dotenv()  # Load credentials from .env

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY")

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="AAPL Stock Forecast", layout="wide")
st.title("AAPL Stock Price Forecast (~Next Few Hours)")

st.markdown("""
This dashboard uses **5-minute interval live AAPL data (last 1 month)** stored in MySQL,
trained with an **LSTM model**, and predicts the **next N periods (~next few hours)**.
""")

# ------------------ FETCH LIVE DATA ------------------
def fetch_live_data():
    """Fetch latest AAPL 5-min interval data from Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": "AAPL",
        "interval": "5min",
        "outputsize": "compact",  # last 100 data points
        "apikey": ALPHA_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json().get("Time Series (5min)", {})

        if not data:
            st.warning("API limit reached or no data returned.")
            return None

        df = pd.DataFrame(data).T
        df = df.rename(columns={"4. close": "close"})
        df["date"] = pd.to_datetime(df.index)
        df["close"] = df["close"].astype(float)
        df = df[["date", "close"]].sort_values("date")
        return df

    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return None

# ------------------ DATABASE CONNECTION ------------------
def connect_db():
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        return connection
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

# ------------------ UPDATE MYSQL WITH LIVE DATA ------------------
def update_mysql_with_live_data(connection, new_data):
    """Insert only new records into MySQL table."""
    try:
        cursor = connection.cursor()
        for _, row in new_data.iterrows():
            cursor.execute("""
                INSERT IGNORE INTO aapl_stock (date, close)
                VALUES (%s, %s)
            """, (row["date"], row["close"]))
        connection.commit()
        cursor.close()
        st.success("MySQL table updated with latest data!")
    except Exception as e:
        st.error(f"Failed to update database: {e}")

# ------------------ REFRESH BUTTON ------------------
if 'refresh_trigger' not in st.session_state:
    st.session_state['refresh_trigger'] = 0  # counter to trigger rerun

if st.button("🔄 Refresh Live Data"):
    connection = connect_db()
    new_data = fetch_live_data()
    if new_data is not None:
        update_mysql_with_live_data(connection, new_data)
    connection.close()
    st.session_state['refresh_trigger'] += 1  # triggers automatic rerun

# ------------------ FETCH STORED DATA ------------------
connection = connect_db()
try:
    query = "SELECT date, close FROM aapl_stock ORDER BY date ASC"
    df = pd.read_sql(query, connection, parse_dates=['date'])
    connection.close()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if df.empty:
    st.warning("No stock data found in database.")
    st.stop()

# ------------------ LOAD MODEL & SCALER ------------------
try:
    scaler = load("aapl_scaler.save")
    model = load_model("aapl_lstm_model.h5")
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# ------------------ FORECAST ------------------
df = df.dropna()
scaled_data = scaler.transform(df['close'].values.reshape(-1, 1))
sequence_length = 60
last_60_days = scaled_data[-sequence_length:]

N = st.slider("Select number of future steps to forecast:", 5, 50, 10)
predictions = []
input_seq = last_60_days.copy()

for _ in range(N):
    pred = model.predict(input_seq.reshape(1, sequence_length, 1), verbose=0)
    predictions.append(pred[0, 0])
    input_seq = np.append(input_seq[1:], pred)
    input_seq = input_seq.reshape(sequence_length, 1)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# --- Forecast timestamps (5 min interval)
last_timestamp = df['date'].iloc[-1]
forecast_dates = [last_timestamp + timedelta(minutes=5 * (i + 1)) for i in range(N)]

forecast_df = pd.DataFrame({
    'Timestamp': forecast_dates,
    'Predicted_Close': predictions.flatten()
})

# ------------------ VISUALIZATION ------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['date'], df['close'], label='Historical Prices', color='blue')
ax.plot(forecast_df['Timestamp'], forecast_df['Predicted_Close'],
        label='Forecast', color='red', marker='o')
ax.set_title(f"AAPL Stock Price Forecast - Next {N*5} Minutes (~{round(N*5/60, 2)} hours)")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("Forecasted Prices (Every 5 Minutes)")
st.dataframe(forecast_df.style.format({"Predicted_Close": "{:.2f}"}))

st.caption("🔁 Auto-refresh every few minutes to keep data updated.")
