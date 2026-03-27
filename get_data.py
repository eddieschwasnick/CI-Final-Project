import pandas as pd
import yfinance as yf


# Get market cap data for a specific ticker
def get_market_cap_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Calculate the market cap trend by pulling in historical close prices and then historical shares outstanding
        historical_data = stock.history(period="max")
        historical_data['Market Cap'] = historical_data['Close'] * stock.info['sharesOutstanding']
        return historical_data[['Close', 'Market Cap']]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Get P/E ratio data for a specific ticker
def get_p_e_ratio_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Calculate the P/E ratio trend by pulling in historical close prices and then historical earnings per share (EPS)
        historical_data = stock.history(period="max")
        historical_data['P/E Ratio'] = historical_data['Close'] / stock.info['trailingEps']
        return historical_data[['Close', 'P/E Ratio']]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Get the quarterly revenue growth data for a specific ticker
def get_revenue_growth_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Get the quarterly financials and calculate the revenue growth
        financials = stock.quarterly_financials
        financials = financials.transpose()
        financials['Revenue Growth'] = financials['Total Revenue'].pct_change()
        return financials[['Total Revenue', 'Revenue Growth']]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    




def main():
    ticker = "AAPL"  # Example ticker
    data = get_market_cap_data(ticker)
    if data is not None:
        print(data.head())  # Print the first few rows of the data

# if __name__ == "__main__":
#     main()
