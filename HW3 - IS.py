# Initial imports:
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    print("### Prepare the Data")    

    #! #####################################
    #! ###### Prepare the Data
    #! #####################################

    # Read-in returns dataframe and set index as "Date":
    whale_returns_csv = Path("CSVs/whale_returns.csv")
    whale_returns_df = pd.read_csv(whale_returns_csv, index_col="Date", parse_dates=True, infer_datetime_format=True)

    # Sort index in ascending order:
    whale_returns_df.sort_index(inplace=True)

    # Drop nulls:
    whale_returns_df.dropna(inplace=True)

    # Preview dataframe:
    print("---------------------------------------------------------")
    print("Detect and Remove null values on Whale")
    print("---------------------------------------------------------")
    print(whale_returns_df.head())
    
    # Read Algorithmic returns and set index as "Date"
    algo_returns_csv = Path("CSVs/algo_returns.csv")
    algo_returns_df = pd.read_csv(algo_returns_csv, index_col="Date", parse_dates=True, infer_datetime_format=True)

    # Sort index in ascending order:
    algo_returns_df.sort_index(inplace=True)

    # Drop nulls:
    algo_returns_df.dropna(inplace=True)

    # Preview DataFrame:
    print("---------------------------------------------------------")
    print("Detect and Remove null values on Algo")
    print("---------------------------------------------------------")
    print(algo_returns_df.head())


    # Read S&P500 returns and set index as "Date"
    sp500_history_csv = Path("CSVs/sp500_history.csv")
    sp500_history_df = pd.read_csv(sp500_history_csv, index_col="Date", parse_dates=True, infer_datetime_format=True)

    # Sort index in ascending order:
    sp500_history_df.sort_index(inplace=True)

    # Preview DataFrame:
    print("---------------------------------------------------------")
    print("Detect and Remove null values on S&P 500")
    print("---------------------------------------------------------")
    print(sp500_history_df.head())

    # Remove "$" from values in "Close" column and convert data type from "object" to "float":
    sp500_history_df["Close"] = sp500_history_df["Close"].str.replace("$", "").astype("float")

    # Calculate S&P 500 daily returns using .pct_change() function:
    sp500_daily_returns = sp500_history_df.pct_change()

    # Rename column:    
    sp500_daily_returns.columns = ["S&P 500 Daily Returns"]

    # Drop nulls:    
    sp500_daily_returns.dropna(inplace=True)

    # Preview DataFrame
    print("---------------------------------------------------------")
    print("Remove dollar sign and conver to daily returns")
    print("---------------------------------------------------------")
    print(sp500_daily_returns.head())

    # Concatenate all dataframes into a single dataframe:
    print("---------------------------------------------------------")
    print("Join `Whale Returns`, `Algorithmic Return`, and the `S&P 500 Returns` into a single DataFrame with columns for each portfolio's returns")
    print("---------------------------------------------------------")
    daily_returns_df = pd.concat([whale_returns_df, algo_returns_df, sp500_daily_returns], axis='columns', join='inner')
    print(daily_returns_df.head())

    #! #####################################
    #! ###### Performance Analysis
    #! #####################################
    
    # Plot Daily Returns:
    daily_returns_df.plot(figsize=(10,10), title="Daily Returns")
    plt.margins(x=0)
    plt.show()

    # Plot Cumulative Returns:
    cumulative_returns = (1 + daily_returns_df).cumprod()
    cumulative_returns.plot(figsize = (20,10), title="Cumulative Returns")
    plt.margins(x=0)
    plt.show()

    # Box plot to visually show risk:
    daily_returns_df.plot(kind = "box", figsize = (20,10), title="Portfolio Risk")
    plt.show()

    #! #####################################
    #! ###### Risk Analysis
    #! #####################################

    # Calculate the standard deviation for each portfolio:
    print("---------------------------------------------------------")
    print("Calculate the standard deviation for each portfolio")
    print("---------------------------------------------------------")
    daily_std_df = pd.DataFrame(daily_returns_df.std()).rename(columns = {0:"Standard Deviation"})
    print(daily_std_df)

    # Determine which portfolios are riskier than the S&P 500:
    print("---------------------------------------------------------")
    print("Determine which portfolios are riskier than the S&P 500")
    print("---------------------------------------------------------")
    higher_std = daily_std_df[daily_std_df["Standard Deviation"] > daily_std_df.loc["S&P 500 Daily Returns", "Standard Deviation"]]
    print(higher_std)

    # Calculate the annualized standard deviation (252 trading days):
    annualized_std_df = daily_std_df * np.sqrt(252)

    # Rename "Standard Deviation" column to "Annualized Standard_Deviation"
    annualized_std_df.columns = ["Annualized Standard Deviation"]

    # Make new dataframe with sorted data: 
    annualized_std_df_sorted = annualized_std_df.sort_values("Annualized Standard Deviation", ascending=False)
    print("---------------------------------------------------------")
    print("Calculate the Annulalized Standard Deviation")
    print("---------------------------------------------------------")
    print(annualized_std_df_sorted)


    #! #####################################
    #! ###### Rolling Statistics
    #! #####################################

    # Calculate and plot the rolling standard deviations for all portfolios using a 21-day trading window:
    sp500_rolling_std = daily_returns_df.rolling(window=21).std()
    sp500_rolling_std.plot(figsize = (20,10), title="21-Day Rolling Standard Deviations")
    plt.margins(x=0)
    plt.show()

    # Construct a correlation table:
    print("---------------------------------------------------------")
    print("Calculate the correlation between each stock")
    print("---------------------------------------------------------")
    correlation = daily_returns_df.corr()
    print(correlation)

    # Calculate Rolling Beta (rolling covariance / rolling variance) for a single portfolio compared to the total market:

    # First, calculate rolling covariance:
    rolling_covariance = daily_returns_df['SOROS FUND MANAGEMENT LLC'].rolling(window=60).cov(daily_returns_df['S&P 500 Daily Returns'])

    # Now, calculate rolling variance:
    rolling_variance = daily_returns_df['S&P 500 Daily Returns'].rolling(window=60).var()

    # Finally, calculate and plot rolling beta:
    rolling_beta = rolling_covariance / rolling_variance
    rolling_beta.plot(figsize=(20, 10), title='Rolling 60-Day Beta of Soros Fund')
    plt.margins(x=0)
    plt.show()

    #! #####################################
    #! ###### Rolling Statics Challenge: Exponentially Weighted Average
    #! #####################################

    # Calculate and plot a rolling window using the exponentially weighted moving average:
    rolling_ewm = daily_returns_df.ewm(span = 21, adjust = False).mean()
    (1 + rolling_ewm).cumprod().plot(figsize = (20,10), title="21-Day Exponentially Weighted Moving Averages")
    plt.margins(x=0)
    plt.show()

    #! #####################################
    #! ###### Sharp Ratios
    #! #####################################

    # Calculate annualized Sharpe Ratios:
    annualized_sharpe_ratios = daily_returns_df.mean()*252 / (daily_returns_df.std()*np.sqrt(252))

    # Sort Sharpe Ratios:
    annualized_sharpe_ratios_sorted = annualized_sharpe_ratios.sort_values(ascending=False)
    annualized_sharpe_ratios.plot(kind = "bar", title = "Annualized Sharpe Ratios")
    plt.show()


    #! #####################################
    #! ###### Create a Custom Portfolio
    #! #####################################
    # Read the first stock:
    apple_returns_csv = Path("CSVs/aapl_historical.csv")
    apple_df = pd.read_csv(apple_returns_csv, index_col="Trade DATE", parse_dates=True, infer_datetime_format=True)

    # Drop "Symbol" column:
    apple_df = apple_df.drop('Symbol', axis=1)

    # Rename column to identify "AAPL Close":
    apple_df.columns = ["AAPL Close"]

    # Read the second stock:
    google_returns_csv = Path("CSVs/goog_historical.csv")
    google_df = pd.read_csv(google_returns_csv, index_col="Trade DATE", parse_dates=True, infer_datetime_format=True)

    # Drop "Symbol" column:
    google_df = google_df.drop('Symbol', axis=1)

    # Rename column to identify "GOOG Close":
    google_df.columns = ["GOOG Close"]

    # Read the third stock:
    costco_returns_csv = Path("CSVs/cost_historical.csv")
    costco_df = pd.read_csv(costco_returns_csv, index_col="Trade DATE", parse_dates=True, infer_datetime_format=True)

    # Drop "Symbol" column:
    costco_df = costco_df.drop('Symbol', axis=1)

    # Rename column to identify "COST Close":
    costco_df.columns = ["COST Close"]

    # Concatenate all stocks into a single dataframe:
    my_stocks_df = pd.concat([apple_df, google_df, costco_df], axis='columns', join='inner')

    # Sort new dataframe:
    my_stocks_df.sort_index(ascending=True, inplace=True)

    # Drop Nulls
    my_stocks_df.dropna(inplace=True)

    print("---------------------------------------------------------")
    print("Calculate the portfolio returns")
    print("---------------------------------------------------------")
    print(my_stocks_df.head())

    # Calculate weighted returns for the portfolio, assuming an equal number of shares for each stock:
    weights = [1/3, 1/3, 1/3]
    my_portfolio_returns = my_stocks_df.pct_change().dot(weights)
    my_portfolio_returns.dropna(inplace=True)

    print("---------------------------------------------------------")
    print("Caculate teh weighted returns for your portfolio, assuming equal number of shares per stock")
    print("---------------------------------------------------------")
    print(my_portfolio_returns.head())

    
    # Add Custom Portfolio to the larger dataframe of fund returns:
    total_portfolio_returns_df = pd.concat([my_portfolio_returns, daily_returns_df], axis='columns', join='inner')
    total_portfolio_returns_df.rename(columns = {0:"AAPL/GOOG/COST"}, inplace = True)

    # Drop nulls:
    total_portfolio_returns_df.dropna(inplace=True)
    print("---------------------------------------------------------")
    print("Add your portfolio returns to the DataFrame with the other portfolios")
    print("---------------------------------------------------------")
    print(total_portfolio_returns_df.head())
    
    # Risk:
    total_portfolio_std = pd.DataFrame(total_portfolio_returns_df.std()).rename(columns = {0:"Standard Deviation"})
    total_portfolio_std = total_portfolio_std.sort_values(by='Standard Deviation', ascending=False)
    print("---------------------------------------------------------")
    print("Calculate the Anulalized Starndard Deviation")
    print("---------------------------------------------------------")
    print(total_portfolio_std)

    # Plot Rolling 21-day standard deviations for all portfolios:
    portfolio_rolling_std = total_portfolio_returns_df.rolling(window=21).std()
    portfolio_rolling_std.plot(figsize = (20,10), title="21-Day Rolling Standard Deviations")
    plt.margins(x=0)
    plt.show()

    
    # Construct a correlation table:
    correlation = total_portfolio_returns_df.corr()
    print("---------------------------------------------------------")
    print("Calculate the correlation")
    print("---------------------------------------------------------")
    print(correlation)

    correlation.plot(figsize = (20, 10), title="Correlation")
    plt.show()

    # Rolling Beta:
    rolling_covariance = total_portfolio_returns_df['AAPL/GOOG/COST'].rolling(window=60).cov(total_portfolio_returns_df['S&P 500 Daily Returns'])
    rolling_variance = total_portfolio_returns_df['AAPL/GOOG/COST'].rolling(window=60).var()
    rolling_beta = rolling_covariance / rolling_variance
    rolling_beta.plot(figsize=(20, 10), title='Rolling 60-Day Beta of AAPL/GOOG/COST')
    plt.margins(x=0)
    plt.show()

    # Annualized Sharpe Ratios for custom portfolio:
    my_portfolio_annualized_sharpe_ratios = total_portfolio_returns_df.mean() * 252 / (total_portfolio_returns_df.std() * np.sqrt(252))

    # Sort Sharpe Ratios:
    my_portfolio_annualized_sharpe_ratios_sorted = my_portfolio_annualized_sharpe_ratios.sort_values(ascending=False)

    # Drop nulls:
    my_portfolio_annualized_sharpe_ratios_sorted.dropna(inplace=True)
    print("---------------------------------------------------------")
    print("Calculate the Sharpe Ratios")
    print("---------------------------------------------------------")
    print(my_portfolio_annualized_sharpe_ratios_sorted)

    # Visualize the sharpe ratios as a bar plot:
    my_portfolio_annualized_sharpe_ratios_sorted.plot(kind = "bar", title = "Annualized Sharpe Ratios")
    plt.show()

main()