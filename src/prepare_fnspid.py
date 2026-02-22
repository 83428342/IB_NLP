import os
import pandas as pd
from datasets import load_dataset
import pandas_datareader.data as web
from tqdm import tqdm

def prepare_fnspid_multi_stock(tickers, save_dir="../data"):
    print(f"Loading filtered FNSPID dataset from HuggingFace for {len(tickers)} tickers...")
    try:
        ds = load_dataset("benstaf/FNSPID-nasdaq-100-post2019-1newsperrow", split="train")
        # Keep only rows where Stock_symbol is in our target list
        ds = ds.filter(lambda x: x['Stock_symbol'] in tickers)
        news_df = ds.to_pandas()
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return
        
    print(f"Extracted a total of {len(news_df)} news articles.")
    if len(news_df) == 0:
        print("No news found for any ticker. Exiting.")
        return
        
    if 'Article' not in news_df.columns:
        news_df['Lsa_summary'] = news_df['Lsa_summary'].fillna('')
        news_df['Article'] = news_df['Article_title'].fillna('') + ". " + news_df['Lsa_summary']
        
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce').dt.date
    news_df = news_df.dropna(subset=['Date'])
    
    # Group by both Date AND Stock_symbol so we merge only same-ticker news on that day
    news_df = news_df.groupby(['Date', 'Stock_symbol'], as_index=False).agg(
        {'Article': lambda x: ' [SEP] '.join(x.astype(str))}
    )
    
    all_merged_data = []
    
    for ticker in tqdm(tickers, desc="Processing Tickers"):
        ticker_news = news_df[news_df['Stock_symbol'] == ticker].copy()
        if len(ticker_news) == 0:
            continue
            
        start_date = ticker_news['Date'].min()
        end_date = ticker_news['Date'].max()
        
        try:
            stock_data = web.DataReader(ticker, 'stooq', start_date, end_date)
        except Exception as e:
            print(f"Stooq download failed for {ticker}: {e}")
            continue
            
        stock_data = stock_data.reset_index()
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        
        stock_data['Next_Close'] = stock_data['Close'].shift(-1)
        stock_data['Label'] = (stock_data['Next_Close'].astype(float) >= stock_data['Close'].astype(float)).astype(int)
        stock_data = stock_data.dropna(subset=['Next_Close'])
        
        if 'Adj Close' not in stock_data.columns:
            stock_data['Adj Close'] = stock_data['Close']
            
        # Add Ticker column so we can keep track in the large dataset
        stock_data['Stock_symbol'] = ticker
            
        merged_ticker_df = pd.merge(stock_data, ticker_news, on=['Date', 'Stock_symbol'], how='inner')
        all_merged_data.append(merged_ticker_df)
        
    if not all_merged_data:
        print("No data was successfully merged.")
        return
        
    # Concatenate all tickers into one massive dataframe vertically
    final_dataset = pd.concat(all_merged_data, ignore_index=True)
    
    # Sort by date
    final_dataset = final_dataset.sort_values(['Date', 'Stock_symbol']).reset_index(drop=True)
    
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "FNSPID_MULTI_merged.csv")
    final_dataset.to_csv(out_path, index=False)
    
    print(f"\n=============================================")
    print(f"Saved {len(final_dataset)} total rows to {out_path}.")
    print(f"Tickers included: {final_dataset['Stock_symbol'].unique()}")
    print("Aggregate Label count:\n", final_dataset['Label'].value_counts(normalize=True))
    print(f"=============================================\n")

if __name__ == "__main__":
    # Top 20 heavily traded NASDAQ-100 Tech/Consumer stocks
    target_tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "NFLX", 
        "INTC", "CSCO", "CMCSA", "PEP", "AVGO", "TXN", "ADBE", "QCOM", 
        "AMD", "SBUX", "AMGN", "COST", "PYPL", "GILD", "INTU"
    ]
    prepare_fnspid_multi_stock(target_tickers)
