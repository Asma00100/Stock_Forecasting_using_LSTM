import requests
import pandas as pd
import pymysql
import traceback
from dotenv import load_dotenv
import os

# =============================
# LOAD ENVIRONMENT VARIABLES
# =============================
load_dotenv()  # Load variables from .env

TICKER = 'AAPL'
INTERVAL = '5min'

API_KEY = os.getenv("ALPHA_API_KEY")  # Alpha Vantage API key from .env

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))
TABLE_NAME = 'aapl_stock'

# =============================
# FUNCTION TO FETCH DATA
# =============================
def fetch_data():
    print(f"\n📥 Fetching full {TICKER} stock data ({INTERVAL}) from Alpha Vantage...")
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": TICKER,
            "interval": INTERVAL,
            "apikey": API_KEY,
            "outputsize": "full"
        }

        response = requests.get(url, params=params)
        data_json = response.json()

        key = f"Time Series ({INTERVAL})"
        if key not in data_json:
            print("No data found or API limit reached.")
            print("Response:", data_json)
            return None

        df = pd.DataFrame.from_dict(data_json[key], orient="index")
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.reset_index(inplace=True)
        df = df.rename(columns={'index': 'datetime'})

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"Successfully fetched {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        traceback.print_exc()
        return None

# =============================
# FUNCTION TO STORE DATA IN MYSQL
# =============================
def store_to_mysql(data):
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            connect_timeout=10
        )
        cursor = connection.cursor()

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            date DATETIME PRIMARY KEY,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT
        )
        """)

        for _, row in data.iterrows():
            sql = f"""
            REPLACE INTO {TABLE_NAME}
            (date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                row['datetime'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                int(row['volume'])
            ))

        connection.commit()
        print(f"{len(data)} rows inserted into `{TABLE_NAME}` successfully.")

    except Exception as e:
        print(f"Error inserting into MySQL: {e}")
        traceback.print_exc()

    finally:
        try:
            cursor.close()
            connection.close()
        except:
            pass

# =============================
# MAIN SCRIPT (Run Once)
# =============================
if __name__ == "__main__":
    data = fetch_data()
    if data is not None:
        store_to_mysql(data)
        data.to_csv("AAPL_full_data.csv", index=False)
        print("Data saved locally as 'AAPL_full_data.csv'")
    else:
        print("No data to save or insert.")
