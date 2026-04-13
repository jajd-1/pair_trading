import numpy as np 
import pandas as pd 
import data 
import signal_construction as sc # pyright: ignore[reportMissingImports]
import backtesting as bt # pyright: ignore[reportMissingImports]


#set parameters
                                
tickers = [                     #tickers of assets under consideration; all possible pairs will be assessed for cointegration in data.py if find_pairs = True. Default list consists of 30 ETFs.
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
start_date = '2016-01-01'       #starting date for price data
end_date = '2026-04-10'         #ending date for price data
pvalue_threshold = 0.05         #p-value for Engle-Granger test
pair = ['QQQ', 'XLU']           #choose a single pair for signal construction and backtesting
window = 60                     #rolling window in days for computing the z-score of the spread
entry_threshold = 2.0           #value of z-score above/below zero to trigger entering a short/long position on the spread
exit_threshold = 0.5            #value of z-score above/below zero to trigger exiting a short/long position on the spread
cost_bps = 5.0                  #transaction cost in basis points


#set which parts of the program you want to run 

find_pairs = False       #set to false if you already know the pair you want to look at (i.e. you don't need to search for pairs in data.py)
build_strat = True     #set to false if you only want to search for pairs using data.py
plots1 = True           #set to false if you don't want to see plots of the proposed cointegrated pairs from data.py
plots2 = True           #set to false if you don't want to see plots of the spread, z-score and positions from signal_construction.py
plots3 = True           #set to false if you don't want to see plots of returns and drawdowns from backtesting.py


if find_pairs == True: 

    _, best_pairs, list_of_pairs = data.find_coint_pairs(tickers, start_date, end_date, pvalue_threshold)
    print(best_pairs)
    if plots1 == True: 
        data.run_plots1(list_of_pairs, start_date, end_date)


if build_strat == True:
    prices = data.load_prices(pair, start_date, end_date)
    series1 = prices[pair[0]]
    series2 = prices[pair[1]]
    print('p-value is: ', data.test_coint(series1, series2))
    dfs = sc.build_signal_dataframe(prices, window, entry_threshold, exit_threshold)
    signal_df = dfs[0]

    if plots2 == True:
        sc.run_plots2(signal_df, entry_threshold, exit_threshold) 

    backtest_df = bt.backtest_pair(signal_df, sc.estimate_hedge_ratio(prices)[1], cost_bps)

    if plots3 == True:
        bt.run_plots3(df = backtest_df)


