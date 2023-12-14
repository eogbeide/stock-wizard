import pandas as pd
import plotly.express as px
from prophet import Prophet
from plotly import graph_objs as go
import glob

from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
import os
from datetime import timedelta

yf.pdr_override()

# Load ticker and company name mapping from CSV
ticker_name_df = pd.read_csv('company_ticker_name.csv')
ticker_name_mapping = dict(zip(ticker_name_df['Ticker'], ticker_name_df['Company']))

# Tickers list
ticker_list = ['DLTR','DG','COST','KO','TGT','JNJ','HD','WMT','INAB','CCCC','CADL','ADTX', 'MTCH', 'EA', 'PYPL', 'INTC', 'PFE', 'MRNA', 'VWAPY', 'CRL', 'CRM', 'AFRM', 'MU', 'AMAT', 'DELL', 'HPQ', 'BABA', 'VTWG', 'SPGI', 'STX', 'LABU', 'TSM', 'AMZN', 'BOX', 'AAPL', 'NFLX', 'AMD', 'GME', 'GOOG', 'GUSH', 'LU', 'META', 'MSFT', 'NVDA', 'PLTR', 'SITM', 'SPCE', 'SPY', 'TSLA', 'URI', 'WDC']
today = date.today()

# We can get data by our choice by giving days bracket
start_date = "2021-12-01"
end_date = today.strftime("%Y-%m-%d")  # Use today's date as the end date

# Get yesterday's date
yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

files = []

def getData(ticker):
    print(ticker)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname = ticker + '_' + str(today)
    files.append(dataname)
    SaveData(data, dataname, ticker)

# Create a data folder in your current dir.
def SaveData(df, filename, ticker):
    save_path = os.path.expanduser('~/Documents/data/')
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    df.to_csv(os.path.join(save_path, filename + '.csv'))
    ticker_name_mapping[ticker] = filename  # Store the mapping of ticker to filename

# This loop will iterate over ticker list, will pass one ticker to get data, and save that data as a file.
for tik in ticker_list:
    getData(tik)

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

def interactive_plot_forecasting(df, forecast, title):
    fig = px.line(df, x='date', y=['y', 'yhat'], title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Original Price'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Price'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound'))
    return fig

figs = []
for df, title, ticker in zip(dfs, titles, tickers):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig = interactive_plot_forecasting(df, forecast, title)
    figs.append(fig)

# Display the plots
for fig in figs:
    st.plotly_chart(fig)
