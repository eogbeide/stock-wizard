import pandas as pd
import plotly.express

from prophet import Prophet
from plotly import graph_objs as go
import glob

from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
import os

yf.pdr_override()

# Tickers list
ticker_list = ['MTCH', 'EA', 'PYPL', 'INTC', 'PFE', 'MRNA', 'VWAPY', 'CRL', 'CRM', 'AFRM', 'MU', 'AMAT', 'DELL', 'HPQ', 'BABA', 'VTWG', 'SPGI', 'STX', 'LABU', 'TSM', 'AMZN', 'BOX', 'AAPL', 'NFLX', 'AMD', 'GME', 'GOOG', 'GUSH', 'LU', 'META', 'MSFT', 'NVDA', 'PLTR', 'SITM', 'SPCE', 'SPY', 'TSLA', 'URI', 'WDC']
today = date.today()

# We can get data by our choice by giving days bracket
start_date = "2021-12-01"
end_date = today.strftime("%Y-%m-%d")  # Use today's date as the end date

files = []

def getData(ticker):
    print(ticker)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname = ticker + '_' + str(today)
    files.append(dataname)
    SaveData(data, dataname)

# Create a data folder in your current dir.
def SaveData(df, filename):
    save_path = os.path.expanduser('~/Documents/data/')
    df.to_csv(os.path.join(save_path, filename + '.csv'))

# This loop will iterate over ticker list, will pass one ticker to get data, and save that data as a file.
for tik in ticker_list:
    getData(tik)

# Pull data, train model, and predict
def select_files(files):
    num_files = len(files)

    # Print the list of files
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selected_files = []
    # Prompt the user to choose two files
    for _ in range(2):
        while True:
            try:
                choice = st.sidebar.selectbox("Select a file", range(1, num_files + 1), format_func=lambda x: files[x - 1].split('/')[-1].split('_')[0], key=f"selectbox_{_}")
                selected_file = files[choice - 1]
                selected_files.append(selected_file)
                break
            except IndexError:
                st.sidebar.warning("Invalid choice. Please try again.")

    return selected_files

# the path to your csv file directory
mycsvdir = 'C:/Users/eogbeide/Documents/data'

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
    selected_file = selected_file.replace('data"\"', '')  # Remove the ".data" extension
    ticker = ticker.replace('data"\"', '')  # Remove the ".data" extension
    #titles.append(f'Original Vs Predicted ({ticker})')
    titles.append(f'Chart of Original Vs Predicted for ({ticker})')

def interactive_plot_forecasting(df, forecast, title):
    fig = px.line(df, x='ds', y=['y', 'predicted'], title=title)

    # Get maximum and minimum points
    max_points = df[df['y'] == df['y'].max()]
    min_points = df[df['y'] == df['y'].min()]

    # Add maximum points to the plot
    fig.add_trace(go.Scatter(x=max_points['ds'], y=max_points['y'], mode='markers', name='Maximum'))

    # Add minimum points to the plot
    fig.add_trace(go.Scatter(x=min_points['ds'], y=min_points['y'], mode='markers', name='Minimum'))

    # Add yhat_lower and yhat_upper
    fig.add_trace(go.Scatter(x=df['ds'], y=forecast['yhat_lower'], mode='lines', name='yhat_lower'))
    fig.add_trace(go.Scatter(x=df['ds'], y=forecast['yhat_upper'], mode='lines', name='yhat_upper'))

    st.plotly_chart(fig)

# Append today's date to the titles
today = date.today().strftime("%Y-%m-%d")

# Iterate over the selected files and their corresponding titles
for df, title, ticker in zip(dfs, titles, tickers):
    # Split the data into testing and training datasets
    train = df[df['ds'] <= '10/31/2023']
    test = df[df['ds'] >= '11/01/2023']

    st.title("Major US Stocks Forecast Wizard")
    st.write("")
    st.subheader("The Smart Stock Trend Wiz: $$$")
    st.write({ticker})
    st.write("How to read chart: Below yhat_lower --> buy signal, above yhat_upper --> sell signal")
    #st.write(f"Number of months in train data for {ticker}: {len(train)}")
    #st.write(f"Number of months in test data for {ticker}: {len(test)}")
    st.write(f"Number of days in train data: {len(train)}")
    st.write(f"Number of days in test data: {len(test)}")

    # Initialize Model
    m = Prophet()

    # Create and fit the prophet model to the training data
    m.fit(train)

    # Make predictions
    future = m.make_future_dataframe(periods=93)
    forecast = m.predict(future)
    #st.write("Forecast for", ticker)
    #   st.write(forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']].tail(30))

    # Add predicted values to the original dataframe
    df['predicted'] = forecast['trend']

    # Plot the forecast and the original values for comparison
    interactive_plot_forecasting(df, forecast, f'{title} ({today})')

# Delete existing files
for file in csvfiles:
    os.remove(file)
