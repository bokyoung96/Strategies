import pandas as pd
import time
from tqdm import tqdm
from polygon import RESTClient

client = RESTClient(api_key="YTvumRUE3CjD5khZl8wYKQVVNqypKLvD")

aggs = []
try:
    for i in client.list_aggs(ticker="SPY", 
                              multiplier=1, 
                              timespan="minute", 
                              from_="2024-01-01", 
                              to="2024-03-31",
                              limit=5000):
        aggs.append(i)
        time.sleep(0.1)
        if len(aggs) % 100 == 0:
            print(f"Collected {len(aggs)} records...")
            
except Exception as e:
    print(f"Error after {len(aggs)} records: {e}")

df = pd.DataFrame(aggs)

