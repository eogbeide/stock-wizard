# Function to compute ATR (Average True Range)
def compute_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    atr = true_range.rolling(window=window).mean()
    return atr

# Function to compute Supertrend
def compute_supertrend(data, atr_multiplier=3, atr_window=14):
    atr = compute_atr(data, window=atr_window)
    hl2 = (data['High'] + data['Low']) / 2  # Average of High and Low (HL2)
    
    # Compute Supertrend bands
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)

    # Initialize Supertrend
    supertrend = pd.Series(index=data.index, dtype=float)
    in_uptrend = True  # Initial trend direction

    for i in range(len(data)):
        if i == 0:
            supertrend.iloc[i] = lower_band.iloc[i]
            continue

        # Determine Supertrend value
        if data['Close'].iloc[i] > supertrend.iloc[i-1]:
            in_uptrend = True
        elif data['Close'].iloc[i] < supertrend.iloc[i-1]:
            in_uptrend = False

        if in_uptrend:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])

    return supertrend

# Streamlit app title
st.title("Stock Price Forecasting using SARIMA with EMA, MA, and Supertrend")

# User input for stock ticker using a dropdown menu
ticker = st.selectbox("Select Stock Ticker:", options=['AAPL', 'SPY', 'AMZN', 'TSLA', 'PLTR', 'NVDA', 'JYD', 'META', 'SITM', 'MARA', 'GOOG', 'HOOD', 'UBER', 'DOW', 'AFRM', 'MSFT', 'TSM', 'NFLX'])

# Button to fetch and process data
if st.button("Forecast"):
    # Step 1: Download historical data from Yahoo Finance
    start_date = '2018-01-01'
    end_date = pd.to_datetime("today")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Step 2: Prepare the data
    prices = data['Close']  # Use the closing prices
    prices = prices.asfreq('D')  # Set frequency to daily
    prices.fillna(method='ffill', inplace=True)  # Forward fill to handle missing values

    # Calculate 200-day EMA
    ema_200 = prices.ewm(span=200, adjust=False).mean()

    # Calculate daily moving average (e.g., 30-day)
    moving_average = prices.rolling(window=30).mean()

    # Calculate Bollinger Bands
    lower_band, middle_band, upper_band = compute_bollinger_bands(prices)

    # Calculate Supertrend
    supertrend = compute_supertrend(data)

    # Step 3: Fit the SARIMA model
    order = (1, 1, 1)  # Example values
    seasonal_order = (1, 1, 1, 12)  # Example values for monthly seasonality

    model = SARIMAX(prices, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Step 4: Forecast the next one month (30 days)
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=prices.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_values = forecast.predicted_mean

    # Get confidence intervals
    conf_int = forecast.conf_int()

    # Step 5: Plot historical data, forecast, EMA, daily moving average, Bollinger Bands, and Supertrend
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot price and 200-day EMA
    ax1.set_title(f'{ticker} Price Forecast, EMA, MA, Bollinger Bands, and Supertrend', fontsize=16)
    ax1.plot(prices[-360:], label='Last 12 Months Historical Data', color='blue')  # Last 12 months of historical data
    ax1.plot(ema_200[-360:], label='200-Day EMA', color='green', linestyle='--')  # 200-day EMA for the last 12 months
    ax1.plot(forecast_index, forecast_values, label='1 Month Forecast', color='orange')
    ax1.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)

    # Add daily moving average for the last 12 months
    ax1.plot(moving_average[-360:], label='30-Day Moving Average', color='brown', linestyle='--')

    # Plot Bollinger Bands
    ax1.plot(lower_band[-360:], label='Bollinger Lower Band', color='red', linestyle='--')
    ax1.plot(upper_band[-360:], label='Bollinger Upper Band', color='pink', linestyle='--')

    # Plot Supertrend
    ax1.plot(supertrend[-360:], label='Supertrend', color='purple', linestyle='-')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Create a DataFrame for forecast data including confidence intervals
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Price': forecast_values,
        'Lower Bound': conf_int.iloc[:, 0],
        'Upper Bound': conf_int.iloc[:, 1]
    })

    # Show the forecast data in a table
    st.write(forecast_df)
