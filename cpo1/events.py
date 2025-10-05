import os
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz

KST = pytz.timezone("Asia/Seoul")


class EventLoader:
    def __init__(self, crawled_path="upbit_listings/upbit_listings.parquet"):
        self.path = Path(crawled_path)

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run price.py first.")
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

        if df["event_time"].dt.tz is None:
            df["event_time"] = df["event_time"].dt.tz_localize(
                KST, nonexistent="NaT", ambiguous="NaT")
        else:
            df["event_time"] = df["event_time"].dt.tz_convert(KST)

        return df[["ticker", "event_time"]]


class PriceDataLoader:
    def __init__(self, price_path: str = None):
        if price_path is None:
            base_dir = Path("prices")
            candidates = sorted(base_dir.glob("binance_prices_*.parquet"))
            if not candidates:
                raise FileNotFoundError(
                    "No price parquet found. Run price.py first.")
            self.path = candidates[-1]
        else:
            self.path = Path(price_path)

    def load(self):
        df = pd.read_parquet(self.path)

        df.index = pd.to_datetime(df.index, errors="coerce")
        if df.index.tz is None:
            df.index = df.index.tz_localize(KST)
        else:
            df.index = df.index.tz_convert(KST)

        df = df.apply(pd.to_numeric, errors="coerce")
        return df


class EventStudy:
    def __init__(self, prices: pd.DataFrame, events: pd.DataFrame, window_days=5):
        self.prices = prices
        self.events = events
        self.window = timedelta(days=window_days)
        self.results = []
        self.aligned_returns = []

    def run(self):
        aligned_list = []
        meta = []

        for _, row in self.events.iterrows():
            ticker = f"{row['ticker']}_USDT"
            event_time = row["event_time"]

            if ticker not in self.prices.columns:
                continue

            s = self.prices[ticker].dropna()
            s = s.loc[(s.index >= event_time) & (
                s.index <= event_time + self.window)]
            if s.empty:
                continue

            start_price = s.iloc[0]
            ret = s / start_price - 1.0

            x = (ret.index - event_time).total_seconds() / 86400.0
            ret.index = x

            full_index = np.linspace(0.0, 5.0, 481)
            ret.loc[0.0] = 0.0
            ret = ret.sort_index().reindex(full_index).interpolate(limit_direction="both")

            if (ret.index.max() < 5.0) or (ret.isna().any().any()):
                continue

            aligned_list.append(ret)
            meta.append({
                "ticker": ticker,
                "event_time": event_time,
                "final_return": float(np.ravel(ret.iloc[-1])[0]),
            })

        if not aligned_list:
            print("⚠️ No valid samples found.")
            self.result_df = pd.DataFrame(
                columns=["ticker", "event_time", "final_return"])
            return pd.DataFrame()

        all_returns = pd.concat(aligned_list, axis=1)
        all_returns.columns = [m["ticker"] for m in meta]
        self.aligned_returns = all_returns
        self.result_df = pd.DataFrame(meta)

        mean_ret = all_returns.mean(axis=1)
        max_ret = all_returns.max(axis=1)
        min_ret = all_returns.min(axis=1)

        self.summary_stats = {
            "samples": int(len(meta)),
            "mean_final_ret": float(mean_ret.iloc[-1]),
            "std_final_ret": float(all_returns.iloc[-1].std()),
        }

        self._plot_event_curve(mean_ret, max_ret, min_ret)
        return all_returns

    def _plot_event_curve(self, mean_ret, max_ret, min_ret):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(mean_ret.index, mean_ret, color="blue",
                 label="Mean Cumulative Return")
        plt.axvline(0, color="gray", linestyle="--", lw=1)
        plt.title("Mean Cumulative Return (t=0 Event Date)")
        plt.xlabel("Days Since Event")
        plt.ylabel("Cumulative Return")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(max_ret.index, max_ret, label="Max", color="green")
        plt.plot(min_ret.index, min_ret, label="Min", color="red")
        plt.axvline(0, color="gray", linestyle="--", lw=1)
        plt.title("Max & Min Cumulative Return")
        plt.xlabel("Days Since Event")
        plt.ylabel("Cumulative Return")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def summary(self):
        if not hasattr(self, "summary_stats"):
            print("Run the study first.")
            return None
        print("\n=== Event Study Summary ===")
        for k, v in self.summary_stats.items():
            print(f"{k:>15}: {v}")
        return self.summary_stats

    @property
    def details(self):
        return getattr(self, "result_df", pd.DataFrame())


class EventPipeline:
    def __init__(self):
        self.event_loader = EventLoader()
        self.price_loader = PriceDataLoader()

    def run(self, window_days=5):
        events = self.event_loader.load()
        prices = self.price_loader.load()

        study = EventStudy(prices, events, window_days=window_days)
        returns = study.run()
        summary = study.summary()

        if returns.empty:
            print("No valid results to save.")
            return returns, summary

        result_path = Path("events") / "event_aligned_returns.parquet"
        result_path.parent.mkdir(exist_ok=True)
        returns.to_parquet(result_path)
        print(f"\n✅ Saved aligned returns → {result_path}")

        return returns, summary


if __name__ == "__main__":
    pipeline = EventPipeline()
    returns, summary = pipeline.run(window_days=5)
