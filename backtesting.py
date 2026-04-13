import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import data 
import signal_construction as sc # pyright: ignore[reportMissingImports]

def backtest_pair(signal_df: pd.DataFrame, beta: float, cost_bps: float) -> pd.DataFrame:   
    """Computes (amongst other quantities) cumulative returns and drawdowns, and builds a dataframe with these quantities"""
    y = signal_df.iloc[:, 0]
    x = signal_df.iloc[:, 1]
    position_execution = signal_df['position'].shift(1).fillna(0.0)     #assume we trade at close each day, so our pnl is determined by close prices that day but our change in position is only reflected the next day

    dy = y.diff().fillna(0.0)          #change from previous day's close price
    dx = x.diff().fillna(0.0)

    spread_pnl = dy - beta * dx     #pnl from today's position if long spread (i.e. long one share of y and short beta shares of x)

    gross_exposure = (y.shift(1).abs() + abs(beta * x.shift(1))).fillna(0.0)      #gross exposure (if a position is held) using previous day's close price

    spread_return = pd.Series(0.0, index = signal_df.index)
    mask = gross_exposure > 0
    spread_return.loc[mask] = spread_pnl.loc[mask] / gross_exposure.loc[mask]   #gross return on today's position assuming a position is held

    gross_strategy_return = position_execution * spread_return      #gross return on today's position 

    turnover = abs(position_execution.diff()).fillna(abs(position_execution))   #checks if a position has been entered

    cost = turnover * (cost_bps / 10000.0)      #computes transaction cost if a position has been entered

    net_strategy_return = gross_strategy_return - cost 

    cumulative_return = (1.0 + net_strategy_return).cumprod()

    running_max = cumulative_return.cummax()
    drawdown = cumulative_return / running_max - 1.0

    backtest_df = signal_df.copy()
    backtest_df["dy"] = dy
    backtest_df["dx"] = dx
    backtest_df["position_exec"] = position_execution
    backtest_df["spread_pnl"] = spread_pnl
    backtest_df["gross_exposure"] = gross_exposure
    backtest_df["spread_return"] = spread_return
    backtest_df["gross_strategy_return"] = gross_strategy_return
    backtest_df["turnover"] = turnover
    backtest_df["cost"] = cost
    backtest_df["strategy_return"] = net_strategy_return
    backtest_df["cumulative_return"] = cumulative_return
    backtest_df["drawdown"] = drawdown

    return backtest_df


# -------------- Visual inspection ---------------


def plot_cumulative_return(backtest_df: pd.DataFrame):
    backtest_df["cumulative_return"].plot(figsize=(12, 5))
    plt.title(f"Cumulative return (growth of $1) for pairs trading strategy with {backtest_df.columns[0]} and {backtest_df.columns[1]}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.show()


def plot_drawdown(backtest_df: pd.DataFrame):
    backtest_df["drawdown"].plot(figsize=(12, 4))
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()


def run_plots3(df: pd.DataFrame):
    plot_cumulative_return(df)
    plot_drawdown(df)





