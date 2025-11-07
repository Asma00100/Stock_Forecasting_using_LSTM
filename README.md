AAPL Stock Price Forecasting

Overview
This project predicts AAPL stock prices using a 5-minute interval LSTM model trained on historical data stored in MySQL. Predictions are displayed in a Streamlit dashboard.


Project Flow
- Fetch Data: fetch_data.py collects live AAPL data from Alpha Vantage API.
- Store Data: Data is stored in MySQL for historical reference.
- Train Model: model_train.py trains an LSTM model on historical close prices and saves the model and scaler.
- Forecast & Visualize: app.py uses the trained model to predict future prices and shows them on a Streamlit dashboard with charts.


Setup
- Clone the repo and create a virtual environment.
- Install dependencies using pip install -r requirements.txt.
- Configure .env with your MySQL credentials and Alpha Vantage API key.
- Run the dashboard using streamlit run app.py.


Notes
- .env is used for sensitive info and is not pushed to GitHub.
- Historical data, model, and scaler files are included for easy setup.