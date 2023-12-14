import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from plotly import graph_objs as go
from pandas_datareader import data as pdr
from datetime import date, timedelta
import yfinance as yf
import os
import csv

yf.pdr_override()

def read_ticker_company_names():
    ticker_company_dict = {}
    with open('company_ticker_name.csv', 'r', encoding='cp1252', errors="ignore") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ticker_company_dict[row['Ticker']] = row['Company']
    return ticker_company_dict

@st.cache
def get_data(ticker, start_date, end_date):
    ticker_company_dict = read_ticker_company_names()
    company = ticker_company_dict.get(ticker, 'Unknown Company')
    st.write(f"Getting data for {ticker} ({company})")
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    return data

def save_data(df, filename):
    save_path = os.path.expanduser('~/Documents/data/')
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    df.to_csv(os.path.join(save_path, filename + '.csv'))

# Tickers list
ticker_list = ['SHOP','ULTA','FL','LULU','DPZ','SHAK','DPZ','SBUX','ETN','CMI','BAC','T','GE','MCD','GILD','PFE','LLY','MMM','ABT','BMY','SPOT','TWLO','PINS','SNAP','LCID','F','RIVN','ADBE','PATH','ORCL','COIN','ABNB','NIO','DLTR','DG','COST','KO','TGT','JNJ','HD','WMT','INAB','CCCC','CADL','ADTX', 'MTCH', 'EA', 'PYPL', 'INTC', 'PFE', 'MRNA', 'CRL', 'CRM', 'AFRM', 'MU', 'AMAT', 'DELL', 'HPQ', 'BABA', 'VTWG', 'SPGI', 'STX', 'LABU', 'TSM', 'AMZN', 'BOX', 'AAPL', 'NFLX', 'AMD', 'GME', 'GOOG', 'GUSH', 'LU', 'META', 'MSFT', 'NVDA', 'PLTR', 'SITM', 'SPCE', 'SPY', 'TSLA', 'URI', 'WDC']
today = date.today()

# We can get data by our choice by giving days bracket
start_date = "2021-12-01"
end_date = today.strftime("%Y-%m-%d")  # Use today's date as the end date

# Get yesterday's date
yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

files = []

# This loop will iterate over ticker list, will pass one ticker to get data, and save that data as a file.
for ticker in ticker_list:
    data = get_data(ticker, start_date, end_date)
    dataname = ticker + '_' + str(today)
    files.append(dataname)
    save_data(data, dataname)

def select_files(files):
    num_files = len(files)

    selected_files = []
    
    while True:
        try:
            choice = st.sidebar.selectbox(
                "Select Company Ticker",
                range(1, num_files + 1),
                format_func=lambda x: files[x - 1].split('/')[-1].split('_')[0],
                key="selectbox"
            )
            selected_file = files[choice - 1]
            selected_files.append(selected_file)
            break
        except IndexError:
            st.sidebar.warning("Invalid choice. Please try again.")

    return selected_files

# the path to your csv file directory
mycsvdir = os.path.expanduser('~/Documents/data')

# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

# Prompt the user to select two files
selected_files = select_files(csvfiles)

# Read the selected files using pandas
dfs = []
for selected_file in selected_files:
    df = pd.read_csv(selected_file)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df.reset_index(inplace=True, drop=True)
    dfs.append(df)

# Plot the selected files
titles = []
tickers = []
for selected_file in selected_files:
    ticker = selected_file.split('/')[-1.split('_')[0]
    tickers.append(ticker)
    ticker_company_dict = read_ticker_company_names()
    company = ticker_company_dict.get(ticker, 'Unknown Company')
    titles.append(f"{ticker} ({company})")

fig = go.Figure()
for i, df in enumerate(dfs):
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name=titles[i]))

fig.update_layout(
    title="Stock Prices",
    xaxis_title="Date",
    yaxis_title="Closing Price",
    legend_title="Tickers",
    hovermode="x"
)

st.plotly_chart(fig)

# Perform forecasting using Prophet
forecast_days = st.sidebar.slider("Select number of days for forecasting:", 1, 365, 30)

forecast_dates = pd.date_range(end=df['ds'].max(), periods=forecast_days + 1, closed='right')
forecast_df = pd.DataFrame({'ds': forecast_dates[:-1]})

forecasted_dfs = []
for df in dfs:
    model = Prophet()
    model.fit(df)
    forecast = model.predict(forecast_df)
    forecasted_dfs.append(forecast)

# Plot the forecasted prices
forecast_fig = go.Figure()
for i, forecast in enumerate(forecasted_dfs):
    forecast_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name=titles[i]))

forecast_fig.update_layout(
    title="Forecasted Stock Prices",
    xaxis_title="Date",
    yaxis_title="Closing Price",
    legend_title="Tickers",
    hovermode="x"
)

st.plotly_chart(forecast_fig)
