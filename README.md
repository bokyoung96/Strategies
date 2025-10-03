# Strategies

Intraday futures research and crypto listing event strategies in one place. The repository currently hosts:

- A minute-level backtesting framework for Korean futures/ETFs with VWAP band, ladder exit, ATR, and momentum overlays (`btm_backtest.py`, `btm_optimizer.py`).
- `cpo1`: Coin IPO Event Strategy 1 – a crawler that watches Upbit listing notices (with hooks prepared for Binance integration).

## Repository Layout

- `btm_backtest.py` – end-to-end intraday backtester (data load → feature engineering → signal → reporting).
- `btm_optimizer.py` – multiprocessing grid search to tune the ladder-exit parameters.
- `db_getter.py` – helper for turning 1-minute SQLite dumps into parquet files used by the backtester.
- `cpo1/` – coin listing crawler (Upbit REST + parsing utilities).
- `vix_backtest.py`, `vpci_backtest.py` – experimental notebooks converted to scripts for volatility/volume studies.

## Intraday Futures Framework

The intraday engine is built for Korean futures and leveraged ETFs but works with any instrument that has tidy 1-minute bars.

**Pipeline**

1. `Loader` reads parquet files like `DATA/A233740_intra.parquet` and exposes them as attributes (`loader.intra`).
2. `FeatureEngineer` derives ATR, VWAP, rolling sigma curves, and day partitions required by the strategies.
3. Strategy classes (see below) translate engineered features into entries/exits.
4. `Portfolio` tracks executions, PnL, and partial exits (ladder logic).
5. `Backtester` iterates minute bars day-by-day and pipes results into the `Reporter` (performance tables, AUM curve, annotated candlesticks).

**Strategy Variants (plug in via `config['strategy_class']`)**

- `BandVWAPStrategy`: baseline mean-reversion around VWAP-adjusted price bands.
- `LadderExitStrategy`: staged profit taking / stop logic with configurable ladder levels.
- `LadderExitMAStrategy`: MA regime filter layered on top of ladder exits.
- `ATRStopStrategy`: ATR-based dynamic stop / take-profit overlays.
- `RollingCandleStrategy`: multi-timeframe TSI confirmation for breakout strength.

**Running a Backtest**

```bash
python btm_backtest.py
```

Edit one of the sample `config_*` dictionaries at the bottom of the file or pass your own when importing `run_backtest`:

```python
from btm_backtest import run_backtest, LadderExitStrategy

config = {
    "name": "Custom Ladder",
    "strategy_class": LadderExitStrategy,
    "asset_code": "A233740_merged",
    "data_folder": "DATA",
    "start_date": "2022-01-01",
    "end_date": "2024-12-31",
    "rolling_move": 14,
    "trade_freq": 10,
    "initial_aum": 1e8,
    "commission_rate": 0.0002,
    "tax_rate": 0.0,
    "strategy_params": {
        "band_multiplier": 1.3,
        "exit_band_multiplier": 1.0,
        "use_vwap": True,
        "ladder_levels": [
            {"stop": -0.012, "profit": 0.0225, "size": 0.5},
            {"stop": 0.008, "profit": 0.04, "size": 1.0},
        ],
    },
}

reporter, daily_results, trades = run_backtest(config)
```

`reporter` prints stats and draws charts; `daily_results`/`trades` are pandas frames ready for further analysis.

**Data Preparation**
If your raw data is stored in SQLite dumps (`DATA/stock_price(1min)_*.db`), run:

```bash
python db_getter.py
```

This saves `*_intra.parquet` and `*_daily.parquet` files after validating daily aggregates. Point `Loader` to those files via `asset_code`.

**Hyper-parameter Search**
`btm_optimizer.py` performs a grid search (CPU-parallel) over rolling window, trading frequency, and ladder parameters. Adjust `param_grid`, then run:

```bash
python btm_optimizer.py
```

The script prints the best Sharpe on the training window and replays the backtest on the holdout dates.

## Coin IPO Event Strategy (`cpo1`)

This package monitors Upbit listing announcements and extracts structured fields for downstream automation (order routing, market making, etc.).

**Main pieces**

- `session.ApiSession`: shared `requests.Session` with custom headers.
- `upbit.UpbitNoticeCrawler`: crawls the Upbit announcement API, parses tickers/markets, and infers original vs. updated go-live times.
- `models`: dataclasses describing summaries and detailed listing metadata.
- `main.get_upbit_new_listings()`: convenience wrapper returning summaries, details, and a pandas table.

**Quick start**

```bash
python -m cpo1.run
```

or inside Python:

```python
from cpo1.main import get_upbit_new_listings
from cpo1.market import Market

data = get_upbit_new_listings(max_items=30, market_mode=Market.KRW)
print(data["crawled"].head())
```

`Market.ALL` keeps every listing; pass `Market.KRW/BTC/USDT` to filter. Binance hooks will live in the same package once the order workflow is finalized.

## Dependencies

Create a virtual environment (Python 3.10+ recommended) and install:

```bash
pip install pandas numpy scipy seaborn matplotlib tqdm requests beautifulsoup4 lxml polygon-api-client
```

(Some scripts import `polygon`, `seaborn`, or `matplotlib`; omit packages you do not need.)

## Notes & Next Steps

- Ensure the `DATA/` folder stays outside Git history; parquet and DB files can get large.
- `vix_backtest.py` ships with a placeholder Polygon API key—set `POLYGON_API_KEY` in your environment before running live pulls.
- TODO: integrate Binance order books into `cpo1` and wire trading actions once listing alerts fire.

Feel free to open issues or drop quick notes in the configs as the strategies evolve.
