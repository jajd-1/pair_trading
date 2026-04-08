import numpy as np 
import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt 
from itertools import combinations
from statsmodels.tsa.stattools import coint 
import statsmodels.api as sm 


# ----------- Define a function to load prices from yfinance --------------- 

def load_prices(tickers, start = '2016-01-01', end = None):
    data = yf.download(tickers, start = start , end = end, auto_adjust = False, progress = False) 

    if "Adj Close" in data: 
        prices = data["Adj Close"].copy()
    else:
        prices = data["Close"].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()      #if there is only one ticker, prices will be a series rather than dataframe, so we convert it for compatability later

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep = 'first')]   #if there are any duplicate rows, keep only the first
    prices = prices.dropna(how = 'any')   #drop rows with any missing values

    return prices 


# ---------------- Select our ETF universe and find possible pairs -----------------------

etfs = [
    "SPY", "IVV", "VOO", "VTI",
    "QQQ", "VGT", "XLK",
    "IWM", "IJR",
    "EFA", "IEFA", "VEA",
    "EEM", "VWO",
    "VNQ", "XLRE",
    "XLF", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU",
    "TLT", "IEF", "SHY", "TLH",
    "GLD", "IAU", "SLV"
]

def generate_pairs(tickers):
    return list(combinations(tickers, 2))


def test_coint(series1, series2):
    score, pvalue, _ = coint(series1, series2)
    return pvalue 


def find_coint_pairs(tickers, pvalue_threshold = 0.05):
    results = []
    pairs = generate_pairs(tickers)
    prices = load_prices(tickers)

    for ticker1, ticker2 in pairs:
        series1 = prices[ticker1]
        series2 = prices[ticker2]

        try:
            pvalue = test_coint(series1, series2)
            results.append({
                'ticker1': ticker1,
                'ticker2': ticker2,
                'pvalue': pvalue 
            })
        
        except Exception as e:
            print(f'Error testing {ticker1}-{ticker2}: {e}')

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pvalue')

    best = results_df[(0 < results_df['pvalue']) & (results_df['pvalue'] < pvalue_threshold)].reset_index(drop = True)
    best_pairs = best[['ticker1', 'ticker2']].values.tolist()

    return results_df, best, best_pairs


# print(find_coint_pairs(etfs)[1])

if __name__ == "__main__":
    best_pairs = find_coint_pairs(etfs)[2]


# --------------- Visual inspection ----------------

def plot_raw_prices(prices):
    prices.plot(figsize = (12,6))
    plt.title('Adjusted Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc = 'best')
    plt.tight_layout


def plot_normalised_prices(prices):
    normalised = prices/prices.iloc[0]
    normalised.plot(figsize = (12,6))
    plt.title('Normalised Adjusted Close Prices (start = 1)')
    plt.xlabel('Date')
    plt.ylabel('Normalised Price')
    plt.legend(loc = 'best')
    plt.tight_layout()
    

def plot_normalised_price_ratios(prices):
    normalised = prices/prices.iloc[0]
    ratio = normalised.iloc[:, 0]/normalised.iloc[:, 1]
    ratio.plot(figsize=(12, 6))
    plt.title(f"{prices.columns[0]}/{prices.columns[1]} Price Ratio")
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.legend(loc = 'best')
    plt.tight_layout()
    

def plot_normalised_price_scatter(prices):
    normalised = prices/prices.iloc[0]
    plt.figure(figsize = (6,6))
    plt.scatter(normalised.iloc[:, 0], normalised.iloc[:, 1], alpha = 0.5, marker = ".")
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.title(f"{prices.columns[1]} vs {prices.columns[0]} Price Scatter")
    plt.tight_layout()


if __name__ == "__main__":
    prices = load_prices(['GLD', 'IAU'])
    plot_raw_prices(prices)
    plt.show()


# for pair in best_pairs: 
#     prices = load_prices(pair)
#     plot_normalised_prices(prices)
#     plot_normalised_price_ratios(prices)
#     plot_normalised_price_scatter(prices)
#     plt.show()


#pairs where ratio seems to vary around the mean: ['IWM', 'VWO'], ['IJR', 'XLB'], ['IJR', 'XLP']
#some other ratios seem to oscillate around a shallow trend, e.g. ['VTI', 'XLU'], ['QQQ', 'XLU']