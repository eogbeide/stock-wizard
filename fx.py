import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from plotly import graph_objs as go
import glob
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
import os
import sys
import datetime

yf.pdr_override()

# Tickers list
ticker_list = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']

today = date.today()

# Check if today is a weekend (Saturday or Sunday)
if today.weekday() >= 5:
    # Display error message
    error_message = "It is weekend; Check back on Monday"
    print(error_message)
    st.subheader("It is the weekend, check back on Monday")
    sys.exit()

# Continue with the rest of your code
# We can get data of our choice by giving days bracket
start_date = "2021-12-01"
end_date = today.strftime("%Y-%m-%d")  # Use today's date as the end date

# Check if today is a weekend (Saturday or Sunday)
if today.weekday() >= 5:
    # Display error message
    #error_message = "It is weekend; Check back on Monday"
    #print(error_message)
    st.write("It is the weekend, check back on Monday when prices are updated")
    #sys.exit()
else:
    #st.write("Welcome to the Stock Trend Prediction Wizard App") 
    st.write("Welcome to the Smart AI Stock Trend Prediction Wizard by Manny: $$$")
    
files = []

# Get yesterday's date
yesterday = today - datetime.timedelta(days=1)

def getData(ticker):
    print(ticker)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname = ticker + '_' + str(today)
    files.append(dataname)
    SaveData(data, dataname)


# Create a data folder in your current dir.
def SaveData(df, filename):
    save_path = os.path.expanduser('~/Documents/data/')
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, filename + '.csv'))


# This loop will iterate over the ticker list, will pass one ticker to get data, and save that data as a file.
for tik in ticker_list:
    getData(tik)


# Pull data, train model, and predict
def select_files(files):
    if not files:
        return []

    num_files = len(files)

    # Print the list of files
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selected_files = []
    # Prompt the user to choose two files
    for _ in range(2):
        while True:
            try:
                choice = st.sidebar.selectbox("Select a file", range(1, num_files + 1),
                                              format_func=lambda x: files[x - 1].split('/')[-1].split('_')[0].split('.')[0], key=f"selectbox_{_}")
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
    # titles.append(f'Original Vs Predicted ({ticker})')
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

    st.title("Major FX Pairs Forecast Wizard")
    st.write("")
    st.subheader("The Smart Forex Trend Wiz: $$$")
    st.write(ticker)
    st.write("How to read the chart: Below yhat_lower --> buy signal, above yhat_upper --> sell signal")
    st.write(f"Number of days in train data: {len(train)}")
    st.write(f"Number of days in test data: {len(test)}")

    # Initialize Model
    m = Prophet()

    # Create and fit the prophet model to the training data
    m.fit(train)

    # Make predictions
    future = m.make_future_dataframe(periods=93)
    forecast = m.predict(future)

    # Add predicted values to the original dataframe
    df['predicted'] = forecast['trend']

    # Plot the forecast and the original values for comparison
    interactive_plot_forecasting(df, forecast, f'{title} ({today})')

# Extract today's forecast values
    today_forecast = forecast[forecast['ds'] == today]

    # Get today's yhat, yhat_lower, and yhat_upper values
    today_yhat = round(today_forecast['yhat'].values[0],2)
    today_yhat_lower = round(today_forecast['yhat_lower'].values[0],2)
    today_yhat_upper = round(today_forecast['yhat_upper'].values[0],2)

    # Get today's date as a datetime.date object
    today = datetime.date.today()

    # Get yesterday's date
    yesterday = today - datetime.timedelta(days=1)

    # Check if yesterday's date falls on a weekend
    if yesterday.weekday() >= 5:
        # Display message for weekend
        error_message = "Yesterday's actual price is unavailable on weekends"
        print(error_message)
        st.write("- Yesterday's actual price is unavailable on weekends")
    else:
        # Continue with the rest of your code
        print("Yesterday's actual price is available")
        # Get yesterday's actual price
        yesterday_actual_price = round(df[df['ds'] == yesterday]['y'].values[0],2)
        st.write("- Yesterday's Price: ", yesterday_actual_price)
        

    # Check if yesterday's actual price exists
    #st.subheader("Yesterday's Closing Price:")
    #if yesterday in df['ds'].values:
        #yesterday_actual_price = df[df['ds'] == yesterday]['y'].values[0]

    # Display today's forecast values
    #if yesterday_actual_price is not None:
        #st.write("- Yesterday's Price: ", yesterday_actual_price)
    #else:
        #st.write("- Yesterday's Price is not available")
    st.subheader("Current Forecast Price Confidence Intervals:")
    st.write("- yhat_lower: ", today_yhat_lower)
    st.write("- yhat: ", today_yhat)
    st.write("- yhat_upper: ", today_yhat_upper)
    
# Delete existing files
#for file in csvfiles:
    #os.remove(file.replace('\\', '/'))

# Display selected ticker information
#st.write("Selected Ticker Information:")
# st.write(selected_ticker_info)
st.subheader("Other Stats")
st.write(" - 50-Day Average: ", selected_ticker_info['fiftyDayAverage'])
st.write(" - 200-Day Average: ", selected_ticker_info['twoHundredDayAverage'])
#st.write(" - beta: ")
if 'beta' in selected_ticker_info:
    st.write(" - beta:", selected_ticker_info['beta'])
else:
    st.write(" - Beta Not Available")
st.subheader("About Company")
if 'longBusinessSummary' in selected_ticker_info:
    st.write(selected_ticker_info['longBusinessSummary'])
else:
    st.write("Not Available")
