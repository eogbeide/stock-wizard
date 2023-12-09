import os
import pandas as pd
import plotly.express as px
from prophet import Prophet
from plotly import graph_objs as go
import glob

from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf

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
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
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
                choice = int(input(f"Select a file (1-{num_files}): "))
                if 1 <= choice <= num_files:
                    selected_file = files[choice - 1]
                    selected_files.append(selected_file)
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    return selected_files

# the path to your csv file directory
mycsvdir = 'Documents/data'

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
for selected_file in selected_files:
    selected_file = selected_file.replace('Documents/data/', '')  # Remove the "Documents/data/" part
    selected_file = selected_file.replace('.csv', '')  # Remove the ".csv" part
    titles.append(f'Original Vs Predicted ({selected_file})')

def interactive_plot_forecasting(df, title):
    fig = px.line(df, x='ds', y=['y', 'predicted'], title=title)

    # Get maximum and minimum points
    max_points = df[df['y'] == df['y'].max()]
    min_points = df[df['y'] == df['y'].min()]

    # Add maximum points to the plot
    fig.add_trace(go.Scatter(x=max_points['ds'], y=max_points['y'], mode='markers', name='Maximum'))

    # Add minimum points to the plot
    fig.add_trace(go.Scatter(x=min_points['ds'], y=min_points['y'], mode='markers', name='Minimum'))

    # Add yhat_lower and yhat_upper to the plot
    fig.add_trace(go.Scatter(x=df['ds'], y=forecast['yhat_lower'], mode='lines', name='yhat_lower'))
    fig.add_trace(go.Scatter(x=df['ds'], y=forecast['yhat_upper'], mode='lines', name='yhat_upper'))

    fig.show()

# Append