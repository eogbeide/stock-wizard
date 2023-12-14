import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from plotly import graph_objs as go
import glob
from pandas_datareader import data as pdr
from datetime import date, timedelta
import yfinance as yf
import os

yf.pdr_override()

# Tickers list
ticker_list = ['DLTR', 'DG', 'COST', 'KO', 'TGT', 'JNJ', 'HD', 'WMT', 'INAB', 'CCCC', 'CADL', 'ADTX', 'MTCH', 'EA',
               'PYPL', 'INTC', 'PFE', 'MRNA', 'VWAPY', 'CRL', 'CRM', 'AFRM', 'MU', 'AMAT', 'DELL', 'HPQ', 'BABA',
               'VTWG', 'SPGI', 'STX', 'LABU', 'TSM', 'AMZN', 'BOX', 'AAPL', 'NFLX', 'AMD', 'GME', 'GOOG', 'GUSH',
               'LU', 'META', 'MSFT', 'NVDA', 'PLTR', 'SITM', 'SPCE', 'SPY', 'TSLA', 'URI', 'WDC']
today = date.today()

# We can get data by our choice by giving days bracket
start_date = "2021-12-01"
end_date = today.strftime("%Y-%m-%d")  # Use today's date as the end date

# Get yesterday's date
yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

files = []


def get_data(ticker):
    print(ticker)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname = ticker + '_' + str(today)
    files.append(dataname)
    save_data(data, dataname)


# Create a data folder in your current dir.
def save_data(df, filename):
    save_path = os.path.expanduser('~/Documents/data/')
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    df.to_csv(os.path.join(save_path, filename + '.csv'))


# This loop will iterate over ticker list, will pass one ticker to get data, and save that data as a file.
for ticker in ticker_list:
    get_data(ticker)

# Pull data, train model, and predict
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
    ticker = selected_file.split('/')[-1].split('_')[0]
    tickers.append(ticker)
    selected_file = selected_file.replace(mycsvdir + '/', '')  # Remove the directory path
    selected_file = selected_file.replace('.csv', '')  # Remove the ".csv" extension
    ticker = ticker.replace('.data', '')  # Remove the ".data" extension
    titles.append(f'Chart of Original Price (y)   Vs   Predicted Price for ({ticker})')

# Append today's date to the titles
today = date.today().strftime("%Y-%m-%d")


def interactive_plot_forecasting(df, forecast, title):
    fig = px.line(df, x='ds', y=['y', 'predictedy'], title=title)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', name='Lower Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', name='Upper Bound'))
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)


for i in range(len(dfs)):
    m = Prophet(daily_seasonality=True)
    m.fit(dfs[i])
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    title = titles[i]
    interactive_plot_forecasting(dfs[i], forecast, title)

# Look up and print the company name from the company_ticker_name.csv file
company_ticker_name = pd.read_csv('company_ticker_name.csv')
for ticker in tickers:
    company_name = company_ticker_name[company_ticker_name['Ticker'] == ticker]['Name'].values[0]
    st.write(f"Company Name: {company_name}")
