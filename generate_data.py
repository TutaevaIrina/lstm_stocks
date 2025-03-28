import yfinance as yf
import pandas as pd
from tti.indicators import RelativeStrengthIndex, RateOfChange, OnBalanceVolume

# Run this file if the Data folder only contains nyse_tickers.txt to generate all Stock Data

if __name__ == "__main__":
    # Load tickers from the file
    with open('Data/nyse_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Date range
    start_date = '2019-11-15'
    end_date = '2024-11-12'
    min_required_date = pd.Timestamp('2020-01-01').tz_localize(None)  # Ensure timezone-naive for checking later


    # Download, calculate indicators, reorder columns, and save data for each ticker
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)

        # Ensure df index is timezone-naive
        df.index = df.index.tz_localize(None)

        # Check if data is available, complete, and starts from or before the minimum required date
        if not df.empty and not df.isnull().values.any() and df.index[0] <= min_required_date:
            # Flatten any MultiIndex columns and remove ticker suffix for uniformity
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).split('_')[0] for col in df.columns.values]

            # Pct_Gain -  daily Percentage Gain as a ratio (no scaling by 100)
            # Prozentuale tägliche Kursveränderung
            df['Pct_Gain'] = df['Close'].pct_change().fillna(0)

            # Der Preis und Volumen beim Börsenhandel in Beziehung setzt
            # OBV - On Balance Value - Ratio (today’s OBV / yesterday’s OBV)
            obv_indicator = OnBalanceVolume(input_data=df[['Close', 'Volume']])
            df['OBV_Ratio'] = obv_indicator.getTiIndicator()

            # Calculate 40-day MA - 40-Day Moving Average to current price ratio - and adjust by subtracting 1
            # langfristig
            df['40D_MA_to_Price_Ratio'] = df['Close'] / df['Close'].rolling(window=40).mean() - 1

            # Calculate 5-day MA -5-Day Moving Average to current price ratio - and adjust by subtracting 1
            # kurzfristig
            df['5D_MA_to_Price_Ratio'] = df['Close'] / df['Close'].rolling(window=5).mean() - 1

            # Relative Strength Index 14-day 
            # Zeigt ob die Kurse steigen, fallen oder neutral sind; Überkauft > 70, überverkauft < 30
            rsi_indicator = RelativeStrengthIndex(input_data=df[['Close']])
            df['RSI'] = rsi_indicator.getTiIndicator()


            # ROC - Rate of change
            # Misst die Geschwindigkeit der Preisveränderung
            roc_indicator = RateOfChange(input_data=df[['Close']], period=14)
            df['ROC'] = roc_indicator.getTiIndicator()
            
            indicators = ['Pct_Gain', 'OBV_Ratio', '40D_MA_to_Price_Ratio',
                          '5D_MA_to_Price_Ratio', 'RSI', 'ROC']
            other_columns = [col for col in df.columns if col not in indicators]
            df = df[indicators + other_columns]

            # Save to CSV file with ticker name in the 'Data' directory
            filename = f"Data/{ticker}_data.csv"
            df = df.drop(columns=['Adj Close', 'High', 'Low', 'Open', 'Volume'])
            df.to_csv(filename)
            print(f"Data COMPLETE for {ticker} with indicators saved to {filename}")
        else:
            print(f"Data for {ticker} is incomplete or does not start before 2020; skipping.")

    print("Data download and processing complete.")
