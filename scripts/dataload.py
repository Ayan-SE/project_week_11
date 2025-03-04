import yfinance as yf
import pandas as pd

def get_financial_data(tickers, start_date, end_date):
    """
    Retrieves historical financial data from Yahoo Finance.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

