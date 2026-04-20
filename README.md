# Pairs trading

This project is a research pipeline for testing a cointegration-based pairs trading strategy on market price data. More details on the pipeline can be found in the file descriptions below, and we also include a section explaining the required background knowledge to understand the strategy. 


## File summaries

`data.py` Loads and cleans adjusted price data for a user-specified selection of assets (provided as a list of ticker symbols) using the yfinance module (an open source tool that uses Yahoo Finance's publicly available APIs). Given the list of assets, we run the Engle–Granger cointegration test on all possible pairs and retain those pairs for which the null hypothesis of no cointegration can be rejected with a user-specified p-value. We also produce plots relating to these pairs to further help identify economically plausible pairs.

`signal_construction.py` For a given pair of assets we estimate their hedge ratio over a given date range by regressing one price series on the other. We use this to construct a dynamic hedge ratio and corresponding residual spread which is updated daily and based on price data from some fixed window of time (typically a few years). This spread is then standardised to produce an adapative rolling z-score, which is used to generate trading signals as follows: when there are large deviations above the equilibrium we go short on the spread, when there are large deviations below the equilibrium we go long on the spread, and when the prices returns to near the equilibrium we close our position.

`backtesting.py` We backtest the above trading strategy using out-of-sample historical data, incorporating both entry/exit costs from entering/exiting positions *and* rebalancing costs arising from the evolving hedge ratio when computing net returns.

`evaluation.py` A more detailed analysis on the performance of our strategy is carried out. We compute various metrics on performance such as total returns, annualised returns, annualised volatility, annualised Sharpe ratio and maximum drawdown, we provide data pertaining to individual trades (e.g. returns and holding period), and we also provide further statistics such as trade count, hit rates, payoff ratios etc. 

`main.py` Compiles the above into a clean pipeline, with the option to bypass pair selection from data.py if the user already has a pair of assets in mind. Various parameters should be set here, including various dates, windows for computing OLS coefficients and z-scores, entry/exit thresholds, trading costs and the risk-free rate used in computing the Sharpe ratio. 


## Assumptions

We make some simplifying assumptions in creating and evaluating our strategy, including:

- Trades only occur at close
- There are no capital constraints
- Perfect shorting (i.e. no constraints or fees on borrowing, and no recalls)
- No financing costs or margin requirements
- No bid-ask spread or slippage (i.e. no market impact from making the proposed trades)
- Perfect liquidity
- No stop-loss or risk controls (i.e. we can hold our positions until exit thresholds are reached)


## Background and methodology 

Rather than trying to predict the price movement of a particular asset, a pairs trading strategy attempts to predict the relative movement of two cointegrated assets. Roughly speaking, two assets are said to be _cointegrated_ if some linear combination of their prices exhibits a long-term, stable equilibrium relationship. One may then build a trading strategy based on deviations around this equilibrium. 

More precisely, consider two time series $x_t$ and $y_t$ representing daily price data for two assets $X$ and $Y$. The standard Engle-Granger test assumes that both $x_t$ and $y_t$ are integrated to order one, denoted $I(1)$, meaning that they are non-stationary but their first differences ($\Delta x_t = x_t - x_{t-1}$ and $\Delta y_t = y_t - y_{t-1}$) are stationary. One way to check this is to test for a unit root using e.g. the ADF test, although we won't explain this here. The Engle-Granger method then regresses one asset on the other using ordinary least squares, yielding

$$y_t = \alpha + \beta x_t + \epsilon_t$$

for some $\alpha,\beta\in\mathbb{R}$. We call $\epsilon_t$ the _residual_ (or the _spread_ in our pairs trading context). The time series $x_t$ and $y_t$ are cointegrated if the residual $\epsilon_t$ is itself stationary, which can be tested using a standard t-statistic under the null hypothesis of no cointegration. We use `statsmodels.coint` on a pair of time series, which returns the t-statistic and associated p-value. 

Suppose we have now decided that two assets $X$ and $Y$ are likely cointegrated. We use `statsmodels.OLS` to estimate the coefficients $\alpha$ and $\beta$ in the above, from which we can read of the spread:

$$\epsilon_t = y_t - \alpha - \beta x_t.$$

In our pairs trading context, we refer to $\beta$ as the _hedge ratio_. In fact, our code incorporates a dynamic hedge ratio using rolling regression, whereby the coefficients $\alpha$ and $\beta$ are continually updated using some fixed window size of past data. The hope is that this additional flexibility reflect changing correlations and volatilities in the long term. We can then compute the adaptive z-score $Z_N$ of the spread using the rolling mean $\mu_N$ and rolling standard deviation $\sigma_N$ from the previous $N$-day window:

$$Z_N = \frac{\epsilon_t - \mu_N}{\sigma_N}.$$

(The word 'adaptive' here refers to the fact that the last $N$ residuals are computed using their respective $\beta$s, rather than using the current day's $\beta$.) This provides a normalised measure of how much the spread has deviated from its equilibrium. In what follows we assume $N$ is fixed and denote $Z_N$ by $Z$. 

We are now in a position build our strategy: if the z-score becomes high (e.g. above 2), then this indicates that $Y$ is overpriced relative to $X$, in which case we go short on the spread (meaning we go short on 1 share of $Y$ and long on $\beta$ shares of $X$). If the z-score becomes low (e.g. below -2), then this indicates that $Y$ is underpriced relative to $X$, in which case we go long on the spread (meaning we go long on 1 share of $Y$ and short on $\beta$ shares of $X$). If the z-score returns close to zero (e.g. above -0.5 while we are long on the spread, or below 0.5 while we are short on the spread), then we close our position (sell our long positions and buy back our short ones). 

We label our position as +1 if we're currently long on the spread, -1 if we're currently short on the spread, and 0 if our position is flat (i.e. we have no open positions). Let $z_{\text{entry}}$ denote a fixed entry threshold and $z_{\text{exit}}$ a fixed exit threshold. We use the following rules:

- If on day $t$ our position is 0 and at close it holds that $Z < -z_{\text{entry}}$, then we enter a long spread position using the close price and enter our position as +1 from day $t+1$.
- If on day $t$ our position is 0 and at close it holds that $Z > z_{\text{entry}}$, then we enter a short spread position using the close price and enter our position as -1 from day $t+1$.
- If on day $t$ our position is +1 and at close it holds that $Z > -z_{\text{exit}}$, then we exit our position using the close price. If at close it further holds that $Z > z_{\text{entry}}$, then we enter a short spread position using the close price and enter our position as -1 from day $t+1$; otherwise, we enter our position as 0 from day $t+1$.
- If on day $t$ our position is -1 and at close it holds that $Z < z_{\text{exit}}$, then we exit our position using the close price. If at close it further holds that $Z < -z_{\text{entry}}$, then we enter a long spread position using the close price and enter our position as +1 from day $t+1$; otherwise, we enter our position as 0 from day $t+1$.

We note that position reversals (jumping from +1 to -1 or vice versa) should be relatively uncommon for genuinely cointegrated assets and reasonable entry/exit thresholds, but these possibilities must be considered nonetheless and introduce some additional complexity to the code.  

The next step is to backtest the strategy. The gross return on day $t$ from a long (+) or short (-) position is 

$$\pm\ \frac{\Delta y_t - \beta_{t-1}\Delta x_t}{|y_{t-1}| + |\beta x_{t-1}|},$$

where $\beta_{t-1}$ is the hedge ratio computed at close on the previous day. The transcation cost of entering or exiting a trade on day $t$ is specified as a certain number of basis points $C_{bp}$ per unit turnover (i.e. per dollar of trades executed),

$$ \frac{C_{bp}}{10,000},$$

 which is subtracted from the gross return on day $t$. The rebalancing costs, which are typically incurred every day a position is held (with the exception of the day the position is exited), are also subtracted and given by 

 $$ \frac{C_{bp}}{10,000} \cdot \frac{|\Delta \beta_t| x_t}{|y_{t-1}| + |\beta x_{t-1}|}.$$

After subtracting these transaction costs we arrive at the net return $r_t$ for day $t$, and the net returns for a given period is then 

$$ \prod_t (1+r_t) - 1.$$

Finally we evaluate the performance of the strategy. (Already in code, to be added here later)

## An example 

## To do