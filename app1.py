yf.pdr_override()


ticker_list = ['MTCH', 'EA', 'PYPL', 'INTC', 'PFE', 'MRNA', 'VWAPY', 'CRL', 'CRM', 'AFRM', 'MU', 'AMAT', 'DELL', 'HPQ', 'BABA', 'VTWG', 'SPGI', 'STX', 'LABU', 'TSM', 'AMZN', 'BOX', 'AAPL', 'NFLX', 'AMD', 'GME', 'GOOG', 'GUSH', 'LU', 'META', 'MSFT', 'NVDA', 'PLTR', 'SITM', 'SPCE', 'SPY', 'TSLA', 'URI', 'WDC']
today = date.today()


start_date = "2021-12-01"
end_date = today.strftime("%Y-%m-%d")  # Use today's date as the end date

files = []

def getData(ticker):
    print(ticker)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname = ticker + '_' + str(today)
    files.append(dataname)
    SaveData(data, dataname)


def SaveData(df, filename):
    save_path = os.path.expanduser('~/Documents/data/')
    df.to_csv(os.path.join(save_path, filename + '.csv'))


for tik in ticker_list:
    getData(tik)


def select_files(files):
    num_files = len(files)


    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selected_files = []
 
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


mycsvdir = 'C:/Users/eogbeide/Documents/data'


csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))


selected_files = select_files(csvfiles)


dfs = []
for selected_file in selected_files:
    df = pd.read_csv(selected_file)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df.reset_index(inplace=True, drop=True)
    dfs.append(df)


titles = []
tickers = []
for selected_file in selected_files:
    ticker = selected_file.split('/')[-1].split('_')[0]
    tickers.append(ticker)
    selected_file = selected_file.replace(mycsvdir + '/', '')  # Remove the directory path
    selected_file = selected_file.replace('.csv', '')  # Remove the ".csv" extension
    selected_file = selected_file.replace('data"\"', '')  # Remove the ".data" extension
    ticker = ticker.replace('data"\"', '')  # Remove the ".data" extension
  
    titles.append(f'Chart of Original Vs Predicted for ({ticker})')

def interactive_plot_forecasting(df, forecast, title):
    fig = px.line(df, x='ds', y=['y', 'predicted'], title=title)

    
    max_points = df[df['y'] == df['y'].max()]
    min_points = df[df['y'] == df['y'].min()]

    
    fig.add_trace(go.Scatter(x=max_points['ds'], y=max_points['y'], mode='markers', name='Maximum'))

    
    fig.add_trace(go.Scatter(x=min_points['ds'], y=min_points['y'], mode='markers', name='Minimum'))

    
    fig.add_trace(go.Scatter(x=df['ds'], y=forecast['yhat_lower'], mode='lines', name='yhat_lower'))
    fig.add_trace(go.Scatter(x=df['ds'], y=forecast['yhat_upper'], mode='lines', name='yhat_upper'))

    st.plotly_chart(fig)


today = date.today().strftime("%Y-%m-%d")


for df, title, ticker in zip(dfs, titles, tickers):
   
    train = df[df['ds'] <= '10/31/2023']
    test = df[df['ds'] >= '11/01/2023']

    st.title("Major US Stocks Forecast Wizard")
    st.write("")
    st.subheader("The Smart Stock Trend Wiz: $$$")
    st.write({ticker})
    st.write("How to read chart: Below yhat_lower --> buy signal, above yhat_upper --> sell signal")
   
    st.write(f"Number of days in train data: {len(train)}")
    st.write(f"Number of days in test data: {len(test)}")


    m = Prophet()

   
    m.fit(train)

   
    future = m.make_future_dataframe(periods=93)
    forecast = m.predict(future)
    #st.write("Forecast for", ticker)


    
    df['predicted'] = forecast['trend']

    
    interactive_plot_forecasting(df, forecast, f'{title} ({today})')


for file in csvfiles:
    os.remove(file)
