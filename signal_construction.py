import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import data


def estimate_hedge_ratio(prices: pd.DataFrame) -> tuple[float,float]:
    """Regress y (first column) on x (second column) with intercept and return the regression coefficients"""
    y = prices.iloc[:, 0]
    x = prices.iloc[:, 1]

    X = sm.add_constant(x)      #add column of 1s to x to include intercept in OLS
    model = sm.OLS(y,X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    return alpha, beta


def construct_spread(prices: pd.DataFrame, alpha: float, beta: float) -> float:
    """With alpha and beta equal to the regression coefficients, we return the OLS residual (aka spread)"""
    y = prices.iloc[:, 0]
    x = prices.iloc[:, 1]
    
    spread = y - alpha  - beta*x
    spread.name = 'spread'

    return spread 


def compute_zscore(spread: float, window: int) -> tuple[float, float, float]:
    """Computes the z-score of the spread on day T using the rolling mean and rolling standard deviation from the previous window number of days"""
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()

    zscore = (spread - rolling_mean) / rolling_std 
    zscore.name = 'zscore'

    return rolling_mean, rolling_std, zscore 


def generate_positions(zscore: float, entry_threshold: float, exit_threshold: float) -> pd.Series:
    """
    Position convention: +1 = long spread, -1 = short spread, 0 = flat

    Rules:  if flat and zscore < -entry_threshold: enter long spread
            if flat and zscore >  entry_threshold: enter short spread
            if long spread and zscore > -exit_threshold: exit
            if short spread and zscore < exit_threshold: exit
    """
    position = pd.Series(index = zscore.index, dtype = float)
    current_position = 0 

    for t in range(len(zscore)):    #we will later shift positions by one day, i.e. close price on day t determines what we do on day t+1
        z = zscore.iloc[t]

        if pd.isna(z):
            position.iloc[t] = 0 
            continue 

        if current_position == 0:
            if z < -entry_threshold:
                current_position = 1    
            elif z > entry_threshold:
                current_position = -1   
        
        elif current_position == 1:
            if z > -exit_threshold:
                current_position = 0    #should also consider case z > entry_threshold, in which case current position should change to -1?
        
        elif current_position == -1:
            if z < exit_threshold:
                current_position = 0    #similar comment to above 
        
        position.iloc[t] = current_position 

    position.name = 'position'

    return position 

def build_signal_dataframe(prices: pd.DataFrame, window: int, entry_threshold: float , exit_threshold: float) -> tuple[pd.DataFrame, float, float, float]:
    """Combine the above into one dataframe"""
    alpha, beta = estimate_hedge_ratio(prices)
    spread = construct_spread(prices, alpha, beta)
    spread_mean, spread_std, zscore = compute_zscore(spread, window = window) 
    position = generate_positions(zscore, entry_threshold = entry_threshold, exit_threshold = exit_threshold) 

    signal_df = prices.copy()
    signal_df['spread'] = spread
    signal_df['spread_mean'] = spread_mean
    signal_df['spread_std'] = spread_std
    signal_df['zscore'] = zscore
    signal_df['position'] = position

    return signal_df, spread, alpha, beta


# ---------------- Visual inspection ------------------


def plot_spread(signal_df: pd.DataFrame):
    signal_df['spread'].plot(figsize = (12,5))
    plt.title(f'Spread after regressing {signal_df.columns[0]} on {signal_df.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.show()

def plot_zscore(signal_df: pd.DataFrame, entry_threshold: float, exit_threshold: float):
    signal_df['zscore'].plot(figsize = (12,5), label = 'z-score')
    plt.axhline(entry_threshold, linestyle = '--', color = 'red', label = 'Entry thresholds')
    plt.axhline(-entry_threshold, linestyle = '--', color = 'red')
    plt.axhline(exit_threshold, linestyle = ':', color = 'red', label = 'Exit thresholds')
    plt.axhline(-exit_threshold, linestyle = ':', color = 'red')
    plt.axhline(0, linestyle = '-')
    plt.title(f'Rolling z-score of spread after regressing {signal_df.columns[0]} on {signal_df.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Z-score')
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()

def plot_position(signal_df: pd.DataFrame):
    signal_df['position'].plot(figsize=(12, 3))
    plt.title(f'Trading position for {signal_df.columns[0]}/{signal_df.columns[1]} spread')
    plt.xlabel('Date')
    plt.ylabel('Position')
    plt.tight_layout()
    plt.show()

def run_plots2(df: pd.DataFrame, entry_threshold: float, exit_threshold: float):
    plot_spread(df)
    plot_zscore(df, entry_threshold, exit_threshold)
    plot_position(df)