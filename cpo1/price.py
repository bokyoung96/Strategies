import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import pytz
import requests
from tqdm import tqdm

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cpo1.main import get_upbit_new_listings
from cpo1.market import Market

KST = pytz.timezone("Asia/Seoul")
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"


def convert_ts_to_dt(ts: int, tz: Optional[pytz.timezone] = KST) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromtimestamp(ts / 1000, tz=pytz.utc).astimezone(tz)


class ListingLoader:
    def __init__(self, path="upbit_listings/upbit_listings.parquet"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            print(f"Loading listings from {self.path}")
            return pd.read_parquet(self.path)
        print("Fetching listings from API ...")
        data = get_upbit_new_listings(limit=100,
                                      market_mode=Market.ALL,
                                      start_page=1,
                                      end_page=None)
        df = pd.DataFrame(data["crawled"])
        df.to_parquet(self.path, index=False)
        print(f"Saved listings to {self.path}")
        return df


class EventLoader:
    def __init__(self, path="upbit_listings/upbit_listings.parquet"):
        self.path = Path(path)

    def load(self):
        df = pd.read_parquet(self.path)
        if "trade_open_updated" in df.columns:
            df["event_time"] = pd.to_datetime(
                df["trade_open_updated"], errors="coerce")
            fallback = pd.to_datetime(
                df["trade_open_original"], errors="coerce")
            df["event_time"] = df["event_time"].fillna(fallback)
        else:
            df["event_time"] = pd.to_datetime(
                df["trade_open_original"], errors="coerce")
        df = df.dropna(subset=["event_time"])
        df["event_time"] = df["event_time"].dt.tz_localize(
            "Asia/Seoul", nonexistent="NaT", ambiguous="NaT")
        return df[["ticker", "event_time"]]


class BinancePriceFetcher:
    def __init__(self, tickers, timeframe="15m",
                 start="2024-01-01", end=None, max_retry=3):
        self.tickers = tickers
        self.timeframe = timeframe
        self.start = pd.Timestamp(start, tz=KST)
        self.end = pd.Timestamp(end or datetime.now(tz=KST))
        self.max_retry = max_retry
        self.exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self._progress = []

    @property
    def progress(self) -> pd.DataFrame:
        return pd.DataFrame(self._progress)

    def _timeframe_to_ms(self):
        unit = self.timeframe[-1]
        value = int(self.timeframe[:-1])
        if unit == "m":
            return value * 60 * 1000
        if unit == "h":
            return value * 60 * 60 * 1000
        if unit == "d":
            return value * 24 * 60 * 60 * 1000
        raise ValueError(f"Unsupported timeframe: {self.timeframe}")

    def _fetch_via_rest(self, symbol: str, start_ts: int, end_ts: int):
        params = {
            "symbol": symbol,
            "interval": self.timeframe,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        r = requests.get(BINANCE_FAPI, params=params)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "qv", "trades", "tbv", "tqv", "ignore"
        ])
        df["datetime"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
        return df[["datetime", "close"]]

    def fetch_event_window(self, ticker: str, event_time: pd.Timestamp,
                           window_days: int = 5, before_days: int = 1):
        event_utc = event_time.tz_convert("UTC")
        start_ts = int(
            (event_utc - pd.Timedelta(days=before_days)).timestamp() * 1000)
        end_ts = int(
            (event_utc + pd.Timedelta(days=window_days)).timestamp() * 1000)
        symbol = ticker.replace("/", "")
        for attempt in range(self.max_retry):
            try:
                df = self._fetch_via_rest(symbol, start_ts, end_ts)
                if df.empty:
                    self._progress.append({
                        "ticker": ticker,
                        "mode": "event",
                        "rows": 0,
                        "status": "no_data",
                        "event_start": convert_ts_to_dt(start_ts),
                        "event_end": convert_ts_to_dt(end_ts)
                    })
                    return pd.DataFrame()
                self._progress.append({
                    "ticker": ticker,
                    "mode": "event",
                    "rows": len(df),
                    "status": "success",
                    "event_start": convert_ts_to_dt(start_ts),
                    "event_end": convert_ts_to_dt(end_ts),
                    "data_start": df["datetime"].iloc[0],
                    "data_end": df["datetime"].iloc[-1]
                })
                return df
            except Exception as e:
                print(f"[{ticker}] REST fetch error: {e} (attempt {attempt+1})")
                time.sleep(2)
                if attempt == self.max_retry - 1:
                    self._progress.append({
                        "ticker": ticker,
                        "mode": "event",
                        "rows": 0,
                        "status": "fail",
                        "event_start": convert_ts_to_dt(start_ts),
                        "event_end": convert_ts_to_dt(end_ts),
                        "error": str(e)
                    })
        return pd.DataFrame()

    def fetch_ohlcv(self, ticker):
        start_utc = self.start.tz_convert("UTC")
        since = int(start_utc.timestamp() * 1000)
        end_ts = int(self.end.tz_convert("UTC").timestamp() * 1000)
        symbol = ticker.replace("/", "")
        df = self._fetch_via_rest(symbol, since, end_ts)
        if df.empty:
            self._progress.append({
                "ticker": ticker,
                "mode": "full",
                "rows": 0,
                "status": "no_data",
                "start_dt": convert_ts_to_dt(since),
                "end_dt": convert_ts_to_dt(end_ts)
            })
            return pd.DataFrame()
        self._progress.append({
            "ticker": ticker,
            "mode": "full",
            "rows": len(df),
            "status": "success",
            "start_dt": df["datetime"].iloc[0],
            "end_dt": df["datetime"].iloc[-1]
        })
        return df


class PriceStorage:
    def __init__(self, base_dir="prices"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict, timeframe: str, tag="full"):
        ts = datetime.now(KST).strftime("%Y%m%d_%H%M")
        path = self.base_dir / f"binance_prices_{tag}_{timeframe}_{ts}.parquet"
        df = pd.concat({k.replace("/", "_"): v.set_index("datetime")
                       for k, v in data.items()}, axis=1)
        df.to_parquet(path)
        print(f"✅ Saved parquet → {path}")
        return path


class PricePipeline:
    def __init__(self):
        self.listing_loader = ListingLoader()
        self.event_loader = EventLoader()
        self.storage = PriceStorage()
        self._progress = pd.DataFrame()

    @property
    def progress(self):
        return self._progress

    def run(self, timeframe="15m", start="2024-01-01",
            mode="event", window_days=5, before_days=1):
        if mode == "event":
            self.events = self.event_loader.load()
            fetcher = BinancePriceFetcher([], timeframe=timeframe)
            results = {}
            for _, row in tqdm(self.events.iterrows(), total=len(self.events)):
                ticker = f"{row['ticker']}/USDT"
                event_time = row["event_time"]
                print(f"Fetching {ticker} window around {event_time}")
                df = fetcher.fetch_event_window(
                    ticker, event_time,
                    window_days=window_days,
                    before_days=before_days
                )
                if not df.empty:
                    results[ticker] = df
            self._progress = fetcher.progress
            path = self.storage.save(results, timeframe, tag="event")
            return results, self._progress, path
        else:
            df_listings = self.listing_loader.load()
            tickers = [f"{t}/USDT" for t in df_listings["ticker"].unique()]
            fetcher = BinancePriceFetcher(
                tickers, timeframe=timeframe, start=start)
            results = {}
            for t in tqdm(tickers):
                df = fetcher.fetch_ohlcv(t)
                if not df.empty:
                    results[t] = df
            self._progress = fetcher.progress
            path = self.storage.save(results, timeframe, tag="full")
            return results, self._progress, path


if __name__ == "__main__":
    pipeline = PricePipeline()
    prices, progress, path = pipeline.run(
        mode="event", window_days=5, before_days=1)
