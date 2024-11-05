import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='stock_analysis.log'
)
"""
Fetch the list of S&P 500 stock tickers from Wikipedia.
Returns:
    List[str]: A list of stock ticker symbols.
"""
def get_sp500_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    table = tables[0]
    return table['Symbol'].tolist()

"""
Load cached stock data if it exists.
Returns:
    dict: A dictionary containing cached stock data.
"""
def load_cache():
    cache_file = Path("stock_cache.pkl")
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

"""
Save stock data to cache.
Args:
    cache (dict): The stock data cache to save.
"""
def save_cache(cache):
    with open("stock_cache.pkl", 'wb') as f:
        pickle.dump(cache, f)

"""
Fetch stock information from yfinance.
Args:
    ticker (str): Stock ticker symbol.
Returns:
    dict: Stock information dictionary.
"""
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info

"""
Extract relevant financial metrics from stock info.
Args:
    ticker (str): Stock ticker symbol.
    info (dict): Stock information dictionary.
Returns:
    dict: Extracted financial metrics.
"""
def extract_financial_metrics(ticker, info):
    data = {
        'Ticker': ticker,
        'PE_Ratio': info.get('trailingPE', np.nan),
        'PB_Ratio': info.get('priceToBook', np.nan),
        'ROE': info.get('returnOnEquity', np.nan),
        'Debt_to_Equity': info.get('debtToEquity', np.nan),
        'EPS': info.get('trailingEps', np.nan),
        'Market_Cap': info.get('marketCap', np.nan),
        'Dividend_Yield': info.get('dividendYield', np.nan),
        'Beta': info.get('beta', np.nan),
        'Current_Price': info.get('currentPrice', np.nan),
        'Target_Price': info.get('targetMeanPrice', np.nan),
        'Profit_Margin': info.get('profitMargins', np.nan),
        'Operating_Margin': info.get('operatingMargins', np.nan),
        'Quick_Ratio': info.get('quickRatio', np.nan),
        'Revenue_Growth': info.get('revenueGrowth', np.nan),
    }

    # Calculate Upside Potential
    current_price = data['Current_Price']
    target_price = data['Target_Price']
    if pd.notnull(current_price) and pd.notnull(target_price):
        data['Upside_Potential'] = ((target_price / current_price - 1) * 100)
    else:
        data['Upside_Potential'] = np.nan

    return data

"""
Process and retrieve stock data, utilizing cache if available.
Args:
    ticker (str): Stock ticker symbol.
    cache (dict): Cache dictionary.
Returns:
    dict: Processed stock data.
"""
def process_stock_data(ticker, cache):
    cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d')}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        info = get_stock_info(ticker)
        data = extract_financial_metrics(ticker, info)
        cache[cache_key] = data
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

"""
Prepare dataset by fetching stock data in parallel.
Args:
    stock_universe (list): List of stock tickers.
    max_workers (int): Maximum number of threads.
    batch_size (int): Number of stocks to process per batch.
Returns:
    pandas.DataFrame: DataFrame containing stock data.
"""
def prepare_dataset(stock_universe, max_workers=10, batch_size=50):
    data = []
    total = len(stock_universe)
    processed = 0
    cache = load_cache()

    for i in range(0, total, batch_size):
        batch = stock_universe[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(process_stock_data, stock, cache): stock 
                for stock in batch
            }

            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    stock_data = future.result()
                    if stock_data:
                        data.append(stock_data)
                    processed += 1
                    if processed % 10 == 0 or processed == total:
                        print(f"Progress: {processed}/{total} stocks processed")
                except Exception as e:
                    logging.error(f"Error processing {stock}: {e}")

        time.sleep(1)  # Rate limiting

    save_cache(cache)
    df = pd.DataFrame(data)
    df = convert_numeric_columns(df)
    return df

"""
Convert relevant columns in DataFrame to numeric types.
Args:
    df (pandas.DataFrame): DataFrame with stock data.
Returns:
    pandas.DataFrame: DataFrame with numeric columns converted.
"""
def convert_numeric_columns(df):
    numeric_columns = [
        'PE_Ratio', 'PB_Ratio', 'ROE', 'Debt_to_Equity', 'EPS',
        'Dividend_Yield', 'Upside_Potential', 'Revenue_Growth',
        'Profit_Margin', 'Operating_Margin', 'Beta', 'Quick_Ratio',
        'Market_Cap', 'Current_Price', 'Target_Price'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

"""
Score stocks based on financial metrics.
Args:
    data (pandas.DataFrame): DataFrame with stock data.
Returns:
    pandas.DataFrame: DataFrame with added 'Score' column.
"""
def score_stocks(data):
    data = data.copy()
    data['Score'] = 0

    # Valuation metrics
    data.loc[data['PE_Ratio'].between(0, 15, inclusive='both'), 'Score'] += 2
    data.loc[data['PB_Ratio'].between(0, 1.5, inclusive='both'), 'Score'] += 2
    data.loc[data['ROE'] > 0.15, 'Score'] += 2
    data.loc[data['Debt_to_Equity'] < 0.5, 'Score'] += 1
    data.loc[data['EPS'] > 1.0, 'Score'] += 1

    # Growth and momentum
    data.loc[data['Upside_Potential'] > 20, 'Score'] += 2
    data.loc[data['Revenue_Growth'] > 0.1, 'Score'] += 2

    # Profitability
    data.loc[data['Profit_Margin'] > 0.1, 'Score'] += 1
    data.loc[data['Operating_Margin'] > 0.15, 'Score'] += 1

    # Risk metrics
    data.loc[data['Beta'].between(0, 1.5, inclusive='both'), 'Score'] += 1
    data.loc[data['Quick_Ratio'] > 1, 'Score'] += 1

    return data

"""
Generate and save a scatter plot visualization.
Args:
    data (pandas.DataFrame): DataFrame with stock data.
"""
def visualize_results(data):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=data,
        x='PE_Ratio',
        y='ROE',
        size='Market_Cap',
        hue='Score',
        sizes=(50, 500),
        palette='viridis',
        alpha=0.7
    )
    plt.title('Stock Analysis Visualization')
    plt.xlabel('PE Ratio')
    plt.ylabel('Return on Equity (ROE)')
    plt.legend(title='Score', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('stock_analysis.png')
    plt.close()

"""
Display the top N scored stocks.
Args:
    sorted_data (pandas.DataFrame): DataFrame with scored stocks.
    top_n (int): Number of top stocks to display.
"""
def display_top_stocks(sorted_data, top_n=20):
    pd.set_option('display.float_format', '{:.2f}'.format)
    print("\nTop Undervalued Stocks:")
    columns_to_display = [
        'Ticker', 'Score', 'PE_Ratio', 'ROE', 'Upside_Potential',
        'Current_Price', 'Target_Price'
    ]
    top_stocks = sorted_data[columns_to_display].head(top_n).reset_index(drop=True)
    print(top_stocks.to_string(index=False))

"""
Save the analysis results to a CSV file.
Args:
    sorted_data (pandas.DataFrame): DataFrame with scored stocks.
"""
def save_results(sorted_data):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'stock_analysis_{timestamp}.csv'
    sorted_data.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

"""
Main function to execute the stock analysis workflow.
"""
def main():
    start_time = time.time()
    logging.info("Starting stock analysis...")

    stock_universe = get_sp500_stocks()
    data = prepare_dataset(stock_universe)

    if not data.empty:
        scored_data = score_stocks(data)
        sorted_data = scored_data.sort_values(by='Score', ascending=False)

        # Display results
        display_top_stocks(sorted_data, top_n=20)
        save_results(sorted_data)
        visualize_results(sorted_data)

        elapsed_time = time.time() - start_time
        logging.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
    else:
        logging.error("No data available for analysis")
        print("No sufficient data available for analysis.")

if __name__ == "__main__":
    main()