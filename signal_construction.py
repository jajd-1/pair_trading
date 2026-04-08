import numpy as np 
import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt 
from itertools import combinations
from statsmodels.tsa.stattools import coint 
import statsmodels.api as sm 
import data

# tickers = ['GLD', 'IAU']        #just look at one pair for now

# prices = data.load_prices(tickers)

# print(prices.head())


def estimate_hedge_ratio(prices):
    y = prices.iloc[:, 0]
    x = prices.iloc[:, 1]

    X = sm.add_constant(x)      #add column of 1s to x (so we have an intercept)
    model = sm.OLS(y,X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    return alpha, beta

# print(estimate_hedge_ratio(prices))


def construct_spread(prices, alpha, beta):
    y = prices.iloc[:, 0]
    x = prices.iloc[:, 1]
    
    spread = y - alpha  - beta*x
    spread.name = 'spread'

    return spread 

def compute_zscore(spread, window = 60):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()

    zscore = (spread - rolling_mean) / rolling_std 
    zscore.name = 'zscore'

    return rolling_mean, rolling_std, zscore 

def generate_positions(zscore, entry_threshold = 2.0, exit_threshold = 0.5):
    """
    Position convention:
        +1 = long spread
        -1 = short spread
         0 = flat

    Rules:
        if flat and zscore < -entry_threshold: enter long spread
        if flat and zscore >  entry_threshold: enter short spread
        if long spread and zscore > -exit_threshold: exit
        if short spread and zscore < exit_threshold: exit
    """

    position = pd.Series(index = zscore.index, dtype = float)
    
    current_position = 0 

    for t in range(len(zscore)):
        z = zscore.iloc[t]

        if pd.isna(z):
            position.iloc[t] = 0 
            continue 

        if current_position == 0:
            if z < -entry_threshold:
                current_position = 1    #long spread
            elif z > entry_threshold:
                current_position = -1   #short spread 
        
        elif current_position == 1:
            if z > -exit_threshold:
                current_position = 0    #should also consider case z > entry_threshold, in which case current position should change to -1?
        
        elif current_position == -1:
            if z < exit_threshold:
                current_position = 0    #similar comment to above 
        
        position.iloc[t] = current_position 

    position.name = 'position'

    return position 

def build_signal_dataframe(prices, window = 60, entry_threshold = 2.0, exit_threshold = 0.5):
    alpha, beta = estimate_hedge_ratio(prices)
    spread = construct_spread(prices, alpha, beta)
    spread_mean, spread_std, zscore = compute_zscore(spread, window = window) 
    position = generate_positions(zscore, entry_threshold = entry_threshold, exit_threshold = exit_threshold) 

    signal_df = prices.copy()
    signal_df["spread"] = spread
    signal_df["spread_mean"] = spread_mean
    signal_df["spread_std"] = spread_std
    signal_df["zscore"] = zscore
    signal_df["position"] = position

    return signal_df, spread, alpha, beta

# print(build_signal_dataframe(prices)[0])
# print(build_signal_dataframe(prices)[1])

def plot_spread(signal_df):
    signal_df['spread'].plot(figsize = (12,5))
    plt.title('Spread')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend(loc = 'best')
    plt.tight_layout
    plt.show()

def plot_zscore(signal_df, entry_threshold = 2.0, exit_threshold = 0.5):
    signal_df['zscore'].plot(figsize = (12,5))
    plt.axhline(entry_threshold, linestyle = '--', color = 'red')
    plt.axhline(-entry_threshold, linestyle= '--', color = 'red')
    plt.axhline(exit_threshold, linestyle= ':', color = 'red')
    plt.axhline(-exit_threshold, linestyle= ':', color = 'red')
    plt.axhline(0, linestyle="-")
    plt.title("Rolling Z-Score of Spread")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.tight_layout()
    plt.show()

def plot_position(signal_df):
    signal_df["position"].plot(figsize=(12, 3))
    plt.title("Trading Position")
    plt.xlabel("Date")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.show()



# signal_df = build_signal_dataframe(prices)[0]

# plot_spread(signal_df)
# plot_zscore(signal_df)
# plot_position(signal_df)




etfs = data.etfs
best_pairs = data.find_coint_pairs(etfs)[2]

for pair in best_pairs:
    prices = data.load_prices(pair)
    signal_df = build_signal_dataframe(prices)[0]
    plot_spread(signal_df)
    plot_zscore(signal_df)
    plot_position(signal_df)


