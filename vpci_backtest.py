import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


class PreProcess:
    def __init__(self,
                 file_name: str,
                 data_dir: str = None,
                 **kwargs):
        self.file_name = file_name
        self.data_dir = data_dir if data_dir else os.path.join(os.path.dirname(__file__), 'DATA')
    
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, key: str = None) -> dict[str, pd.DataFrame] | pd.DataFrame:
        data = self._get_data()
        return data[key] if key else data

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self._get_data()[key]

    def _get_data(self) -> dict[str, pd.DataFrame]:
        close_path = os.path.join(self.data_dir, f"{self.file_name}_close.parquet")
        volume_path = os.path.join(self.data_dir, f"{self.file_name}_volume.parquet")
        
        if os.path.exists(close_path) and os.path.exists(volume_path):
            return {
                'close': pd.read_parquet(close_path),
                'volume': pd.read_parquet(volume_path)
            }
        else:
            return self._save_data()

    def _save_data(self) -> dict[str, pd.DataFrame]:
        file_path = os.path.join(self.data_dir, f"{self.file_name}.xlsx")
        data = pd.read_excel(file_path,
                             index_col=0,
                             header=7).iloc[6:, :]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'
        
        close_cols = [col for col in data.columns if not col.endswith('.1')]
        volume_cols = [col for col in data.columns if col.endswith('.1')]
        
        close_data = data[close_cols]
        volume_data = data[volume_cols]
        
        close_data.to_parquet(os.path.join(self.data_dir, f"{self.file_name}_close.parquet"))
        volume_data.to_parquet(os.path.join(self.data_dir, f"{self.file_name}_volume.parquet"))
        
        return {'close': close_data, 'volume': volume_data}


class VPCICalculator:
    def __init__(self, close_data: pd.Series, volume_data: pd.Series, 
                 lower_multiplier: float = 2.0, upper_multiplier: float = 2.0):
        self.close_data = close_data
        self.volume_data = volume_data
        self.lower_multiplier = lower_multiplier
        self.upper_multiplier = upper_multiplier
        self._calculate_vpci()
    
    def _calculate_vwma(self, period: int) -> pd.Series:
        price_volume = self.close_data * self.volume_data
        volume_sum = self.volume_data.rolling(window=period, min_periods=period // 2).sum()
        vwma = price_volume.rolling(window=period, min_periods=period // 2).sum() / volume_sum.replace(0, np.nan)
        return vwma.ffill().fillna(0)
    
    def _calculate_ma(self, period: int) -> pd.Series:
        return self.close_data.rolling(window=period, min_periods=period // 2).mean().ffill().fillna(self.close_data.mean())
    
    def _calculate_vol_avg(self, period: int) -> pd.Series:
        return self.volume_data.rolling(window=period, min_periods=period // 2).mean().ffill().fillna(self.volume_data.mean())
    
    def _calculate_vpci(self):
        try:
            self.vwma_20 = self._calculate_vwma(20)
            self.ma_20 = self._calculate_ma(20)
            self.vwma_5 = self._calculate_vwma(5)
            self.ma_5 = self._calculate_ma(5)
            self.vol_avg_13 = self._calculate_vol_avg(13)
            self.vol_avg_52 = self._calculate_vol_avg(52)
            
            self.vpc = self.vwma_20 - self.ma_20
            self.vpr = self.vwma_5 / self.ma_5
            self.vm = self.vol_avg_13 / self.vol_avg_52
            
            self.vpr = self.vpr.replace([np.inf, -np.inf], 1.0)
            self.vm = self.vm.replace([np.inf, -np.inf], 1.0)
            
            self.vpci = (self.vpc * self.vpr * self.vm).fillna(0)
            
            self.vpci_ma = self.vpci.rolling(window=20, min_periods=10).mean().ffill().fillna(0)
            self.vpci_std = self.vpci.rolling(window=20, min_periods=10).std().ffill().fillna(0)
            self.lb_vpci = self.vpci_ma - (self.lower_multiplier * self.vpci_std)
            self.ub_vpci = self.vpci_ma + (self.upper_multiplier * self.vpci_std)
            
        except Exception as e:
            self.vwma_20, self.ma_20, self.vwma_5, self.ma_5, self.vol_avg_13, self.vol_avg_52, self.vpc, self.vpr, self.vm, self.vpci, self.vpci_ma, self.vpci_std, self.lb_vpci, self.ub_vpci = [pd.Series(0, index=self.close_data.index) for _ in range(14)]


class Portfolio:
    def __init__(self, initial_aum: float, commission_rate: float = 0.00015, trailing_stop_pct: float = 0.15):
        self.initial_aum = initial_aum
        self.commission_rate = commission_rate
        self.trailing_stop_pct = trailing_stop_pct
        self.reset()
    
    def reset(self):
        self.aum = self.initial_aum
        self.position = {'direction': 0, 'shares': 0, 'entry_price': 0, 'entry_time': None, 'max_price': 0}
        self.history = pd.DataFrame(columns=['aum', 'pnl'], dtype=float)
        self.trades = []
    
    def record_daily_state(self, date: pd.Timestamp, current_price: float):
        current_aum = self.aum
        if self.position['direction'] != 0:
            unrealized_pnl = (current_price - self.position['entry_price']) * self.position['shares']
            current_aum = (abs(self.position['shares']) * self.position['entry_price']) + unrealized_pnl
            
            if current_price > self.position['max_price']:
                self.position['max_price'] = current_price
        
        self.history.loc[date] = {'aum': float(current_aum), 'pnl': 0.0}
    
    def execute_entry(self, direction: int, price: float, timestamp: pd.Timestamp, trade_amount: float):
        shares = (trade_amount / price) * direction
        self.position = {
            'direction': direction,
            'shares': shares,
            'entry_price': price,
            'entry_time': timestamp,
            'max_price': price
        }
    
    def check_trailing_stop(self, current_price: float) -> bool:
        if self.position['direction'] == 0:
            return False
        
        stop_price = self.position['max_price'] * (1 - self.trailing_stop_pct)
        return current_price <= stop_price
    
    def execute_exit(self, price: float, timestamp: pd.Timestamp, exit_reason: str = "unknown"):
        pos = self.position
        gross_pnl = (price - pos['entry_price']) * pos['shares']
        commission = abs(pos['shares']) * (pos['entry_price'] + price) * self.commission_rate
        net_pnl = gross_pnl - commission
        
        self.aum += net_pnl
        self.history.loc[timestamp, 'pnl'] = float(self.history.loc[timestamp, 'pnl'] + net_pnl)
        self.history.loc[timestamp, 'aum'] = float(self.aum)
        
        self.trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'direction': pos['direction'],
            'exit_reason': exit_reason,
            'shares': abs(pos['shares']),
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'net_return': net_pnl / (abs(pos['shares']) * pos['entry_price']) if pos['shares'] != 0 else 0
        })
        
        self.reset_position()
    
    def reset_position(self):
        self.position = {'direction': 0, 'shares': 0, 'entry_price': 0, 'entry_time': None, 'max_price': 0}


class VPCIStrategy:
    def __init__(self, params: dict):
        self.params = params
        self.strategy_type = params.get('strategy_type', 'vbot') # 'vbot' or 'wbot'
        self.vpci_threshold = params.get('vpci_threshold', 0.4)
        self.last_vbot_date = None
    
    def get_entry_signal(self, row: pd.Series, portfolio) -> tuple[int, str]:
        if portfolio.position['direction'] != 0:
            return 0, "none"
        
        if pd.isna(row['vpci']) or pd.isna(row['lb_vpci']):
            return 0, "none"
        
        vpci_value = row['vpci']
        lb_vpci = row['lb_vpci']
        current_date = row.name
        
        is_signal_condition = vpci_value <= self.vpci_threshold and vpci_value < lb_vpci
        
        if self.strategy_type == 'vbot':
            if is_signal_condition:
                return 1, "vbot"
        
        elif self.strategy_type == 'wbot':
            if is_signal_condition:
                if self.last_vbot_date is None:
                    self.last_vbot_date = current_date
                else:
                    days_since_last = (current_date - self.last_vbot_date).days
                    if 0 < days_since_last <= 20:
                        self.last_vbot_date = None
                        return 1, "wbot"
                    else:
                        self.last_vbot_date = current_date
            
        return 0, "none"
    
    def get_exit_action(self, row: pd.Series, portfolio) -> dict | None:
        if portfolio.position['direction'] == 0:
            return None
        
        if portfolio.check_trailing_stop(row['close']):
            self.last_vbot_date = None
            return {'type': 'full', 'reason': f'trailing_stop_{portfolio.trailing_stop_pct:.3f}'}
        
        return None


class VPCIBandStrategy:
    def __init__(self, params: dict):
        self.params = params
        self.lower_threshold = params.get('lower_threshold', 0.4)
        self.upper_threshold = params.get('upper_threshold', -0.4)
    
    def get_entry_signal(self, row: pd.Series, portfolio) -> tuple[int, str]:
        if portfolio.position['direction'] != 0:
            return 0, "none"
        
        if pd.isna(row['vpci']) or pd.isna(row['lb_vpci']) or pd.isna(row['ub_vpci']):
            return 0, "none"
        
        vpci_value = row['vpci']
        lb_vpci = row['lb_vpci']
        ub_vpci = row['ub_vpci']
        
        if vpci_value <= self.lower_threshold and vpci_value < lb_vpci:
            return 1, "band_long"
        elif vpci_value >= self.upper_threshold and vpci_value > ub_vpci:
            return -1, "band_short"
        
        return 0, "none"
    
    def get_exit_action(self, row: pd.Series, portfolio) -> dict | None:
        if portfolio.position['direction'] == 0:
            return None
        
        if pd.isna(row['vpci']) or pd.isna(row['vpci_ma']):
            return None
        
        vpci_value = row['vpci']
        vpci_ma = row['vpci_ma']
        
        if portfolio.position['direction'] == 1 and vpci_value >= vpci_ma:
            return {'type': 'full', 'reason': 'band_long_exit'}
        elif portfolio.position['direction'] == -1 and vpci_value <= vpci_ma:
            return {'type': 'full', 'reason': 'band_short_exit'}
        
        return None


class VPCIBacktest:
    def __init__(self, preprocess: PreProcess, ticker: str | list[str] = None, **kwargs):
        self.preprocess = preprocess
        self.ticker = ticker
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _prepare_ticker_data(self, ticker: str, start_date: str, end_date: str, 
                           lower_multiplier: float, upper_multiplier: float):
        data = self.preprocess()
        close_data = data['close'][ticker]
        volume_data = data['volume'].get(ticker + '.1')

        if volume_data is None: return None
        
        df = pd.DataFrame({'close': close_data, 'volume': volume_data}).dropna()
        df = df[df['close'] > 0] 

        if len(df) < 52: return None
            
        vpci_calc = VPCICalculator(df['close'], df['volume'], 
                                  lower_multiplier=lower_multiplier, 
                                  upper_multiplier=upper_multiplier)
        
        log_df = pd.DataFrame({
            'close': df['close'],
            'volume': df['volume'],
            'vpci': vpci_calc.vpci,
            'vpci_ma': vpci_calc.vpci_ma,
            'lb_vpci': vpci_calc.lb_vpci,
            'ub_vpci': vpci_calc.ub_vpci,
        })
        
        if start_date:
            log_df = log_df[log_df.index >= pd.to_datetime(start_date)]
        if end_date:
            log_df = log_df[log_df.index <= pd.to_datetime(end_date)]

        if log_df.empty: return None

        return log_df

    def run_single_ticker(self, log_df: pd.DataFrame, 
                         initial_aum: float, commission_rate: float, 
                         trailing_stop_pct: float, strategy_type: str,
                         use_trailing_stop: bool = True,
                         lower_threshold: float = 0.4, upper_threshold: float = -0.4):
        
        portfolio = Portfolio(initial_aum, commission_rate, trailing_stop_pct)
        
        if strategy_type in ['vbot', 'wbot']:
            strategy_params = {'strategy_type': strategy_type, 'vpci_threshold': getattr(self, 'vpci_threshold', 0.4)}
            strategy = VPCIStrategy(strategy_params)
        elif strategy_type == 'band':
            strategy_params = {'lower_threshold': lower_threshold, 'upper_threshold': upper_threshold}
            strategy = VPCIBandStrategy(strategy_params)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        log_df['signal'] = 'none'
        log_df['trade_price'] = np.nan

        for date, row in log_df.iterrows():
            portfolio.record_daily_state(date, row['close'])
            
            if portfolio.position['direction'] != 0:
                action = strategy.get_exit_action(row, portfolio)
                if action:
                    portfolio.execute_exit(row['close'], date, action['reason'])
                    log_df.loc[date, 'signal'] = 'exit'
                    log_df.loc[date, 'trade_price'] = row['close']
                elif use_trailing_stop and portfolio.check_trailing_stop(row['close']):
                    portfolio.execute_exit(row['close'], date, f'trailing_stop_{portfolio.trailing_stop_pct:.3f}')
                    log_df.loc[date, 'signal'] = 'exit'
                    log_df.loc[date, 'trade_price'] = row['close']
            else:
                signal, signal_type = strategy.get_entry_signal(row, portfolio)
                if signal != 0:
                    trade_amount = portfolio.aum
                    portfolio.execute_entry(signal, row['close'], date, trade_amount)
                    log_df.loc[date, 'signal'] = signal_type
                    log_df.loc[date, 'trade_price'] = row['close']
        
        history = portfolio.history.copy()
        if not history.empty:
            strategy_daily_return = history['aum'].pct_change().fillna(0)
            benchmark_daily_return = log_df['close'].pct_change().fillna(0)
            history['daily_return'] = strategy_daily_return - benchmark_daily_return
            history['cumulative_return'] = (history['aum'] / portfolio.initial_aum) - 1
        
        return history, pd.DataFrame(portfolio.trades), log_df
    
    def run(self, initial_aum: float = 1e8, 
            commission_rate: float = 0.0,
            trailing_stop_pct: float = 0.15, 
            start_date: str = None, 
            end_date: str = None,
            strategy_type: str = 'vbot',
            multiplier: float = 2.0,
            vpci_threshold: float = 0.4,
            lower_multiplier: float = None,
            upper_multiplier: float = None,
            lower_threshold: float = 0.4,
            upper_threshold: float = -0.4,
            use_trailing_stop: bool = True):
        
        if lower_multiplier is None:
            lower_multiplier = multiplier
        if upper_multiplier is None:
            upper_multiplier = multiplier
        
        if self.ticker:
            if isinstance(self.ticker, str): tickers_to_run = [self.ticker]
            else: tickers_to_run = self.ticker
        else:
            tickers_to_run = self.preprocess()['close'].columns
        
        self.vpci_threshold = vpci_threshold
        ticker_results = {}
        ticker_performance = []
        
        for ticker in tqdm(tickers_to_run, desc=f"Running {strategy_type.upper()} Strategy"):
            log_df = self._prepare_ticker_data(ticker, start_date, end_date, 
                                             lower_multiplier, upper_multiplier)
            if log_df is None: continue

            try:
                history, trades, final_log_df = self.run_single_ticker(
                    log_df, initial_aum, commission_rate, trailing_stop_pct, strategy_type,
                    use_trailing_stop=use_trailing_stop,
                    lower_threshold=lower_threshold, upper_threshold=upper_threshold
                )
                
                if history.empty: continue

                ticker_results[ticker] = {'history': history, 'trades': trades, 'log': final_log_df}
                
                total_return = history['cumulative_return'].iloc[-1]
                final_aum = history['aum'].iloc[-1]
                total_trades = len(trades)
                win_rate = (trades['net_pnl'] > 0).mean() if not trades.empty else 0
                avg_return = trades['net_return'].mean() if not trades.empty else 0
                max_drawdown = self._calculate_max_drawdown(history['aum'])
                
                ticker_performance.append({
                    'ticker': ticker, 'total_return': total_return, 'total_trades': total_trades,
                    'win_rate': win_rate, 'avg_return': avg_return, 'max_drawdown': max_drawdown,
                    'final_aum': final_aum
                })
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")
                continue
        
        performance_df = pd.DataFrame(ticker_performance)
        if performance_df.empty: print("No tickers were successfully processed.")
        
        return ticker_results, performance_df

    def _calculate_max_drawdown(self, aum_series: pd.Series) -> float:
        if aum_series.empty: return 0
        running_max = aum_series.cummax()
        drawdown = (aum_series - running_max) / running_max
        return drawdown.min()
    
    def plot_ticker_analysis(self, ticker: str, results: dict, strategy_type: str, vpci_threshold: float = 0.4):
        if ticker not in results:
            print(f"No results found for ticker {ticker}.")
            return
            
        result = results[ticker]
        history = result['history']
        log_df = result['log']
        trades = result['trades']
        
        if history.empty:
            print(f"No trading activity for ticker {ticker}.")
            return

        # trailing_stop_pct 추출 로직 개선
        trailing_stop_pct = 0.15  # 기본값
        if not trades.empty:
            first_exit_reason = trades.iloc[0]['exit_reason']
            if 'trailing_stop_' in first_exit_reason:
                try:
                    trailing_stop_pct = float(first_exit_reason.split('_')[-1])
                except ValueError:
                    trailing_stop_pct = 0.15
        
        # --- Benchmark Calculations ---
        benchmark_cum_return = (log_df['close'] / log_df['close'].iloc[0]) - 1
        benchmark_aum = log_df['close']
        benchmark_running_max = benchmark_aum.cummax()
        benchmark_underwater = (benchmark_aum - benchmark_running_max) / benchmark_running_max
        benchmark_max_drawdown = benchmark_underwater.min()
        
        # --- Strategy Calculations ---
        drawdown = self._calculate_max_drawdown(history['aum'])
        running_max = history['aum'].cummax()
        underwater = (history['aum'] - running_max) / running_max
        
        # --- Plotting Setup ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax_daily, ax2, ax3) = plt.subplots(4, 1, figsize=(15, 22), height_ratios=[3, 1, 1.5, 2], sharex=True)
        fig.suptitle(f'{strategy_type.upper()} Strategy Performance - {ticker} (Trailing Stop: {trailing_stop_pct:.1%})', fontsize=16)
        
        # --- AX1: Cumulative Return & Return Difference ---
        ax1.plot(history.index, history['cumulative_return'] * 100, label='Strategy Cumulative Return (%)', color='blue', linewidth=2)
        ax1.plot(benchmark_cum_return.index, benchmark_cum_return * 100, label='Benchmark (Buy & Hold) (%)', color='grey', linestyle='--', linewidth=1.5)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.5)

        entry_signals = log_df[log_df['signal'].isin(['vbot', 'wbot'])]
        exit_signals = log_df[log_df['signal'] == 'exit']
        
        if not entry_signals.empty:
            ax1.scatter(entry_signals.index, history.loc[entry_signals.index, 'cumulative_return'] * 100,
                        color='green', marker='^', s=100, label='Entry Signals', zorder=5)
        
        if not exit_signals.empty:
            ax1.scatter(exit_signals.index, history.loc[exit_signals.index, 'cumulative_return'] * 100,
                        color='red', marker='v', s=100, label='Exit Signals', zorder=5)
        ax1.legend(loc='upper left')

        ax1_twin = ax1.twinx()
        return_diff = (history['cumulative_return'] - benchmark_cum_return) * 100
        ax1_twin.fill_between(return_diff.index, return_diff, 0,
                              where=return_diff >= 0, color='lightgreen', alpha=0.3, interpolate=True, label='Outperformance')
        ax1_twin.fill_between(return_diff.index, return_diff, 0,
                              where=return_diff < 0, color='lightcoral', alpha=0.3, interpolate=True, label='Underperformance')
        ax1_twin.set_ylabel('Strategy vs Benchmark Return Difference (%)', color='gray')
        ax1_twin.tick_params(axis='y', labelcolor='gray')
        ax1_twin.legend(loc='upper right')
        
        # --- AX_DAILY: Daily Returns ---
        daily_returns = history['daily_return'] * 100
        ax_daily.bar(daily_returns.index, daily_returns, color=np.where(daily_returns >= 0, 'g', 'r'), alpha=0.7)
        ax_daily.set_ylabel('Daily Return (%)')
        ax_daily.set_title('Daily Returns (Strategy - Benchmark)', fontsize=12)
        ax_daily.grid(True, alpha=0.5)
        
        # --- AX2: Drawdown ---
        ax2.fill_between(underwater.index, 0, underwater * 100, color='lightblue', alpha=0.3, label=f'Strategy DD (Max: {drawdown:.2%})')
        ax2.plot(benchmark_underwater.index, benchmark_underwater * 100, color='grey', linestyle='--', label=f'Benchmark DD (Max: {benchmark_max_drawdown:.2%})')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title(f'Underwater Plot', fontsize=12)
        ax2.grid(True, alpha=0.5)
        ax2.legend(loc='lower left')
        
        # --- AX3: VPCI Plot ---
        mask = (log_df['vpci'].notna()) & (log_df['lb_vpci'].notna()) & (log_df['ub_vpci'].notna())
        clean_df = log_df[mask]
        
        ax3.plot(clean_df.index, clean_df['vpci'], label='VPCI', color='purple', linewidth=1.5)
        ax3.plot(clean_df.index, clean_df['lb_vpci'], label='LB_VPCI', color='orange', linewidth=1.5)
        ax3.plot(clean_df.index, clean_df['ub_vpci'], label='UB_VPCI', color='cyan', linewidth=1.5)
        ax3.axhline(y=vpci_threshold, color='red', linestyle='--', alpha=0.7, label=f'VPCI Threshold ({vpci_threshold:.1f})')
        
        ax3.set_ylabel('VPCI Value')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()
        
        print(f"\n=== {ticker} ({strategy_type.upper()}) Performance Summary ===")
        total_return = history['cumulative_return'].iloc[-1]
        print(f"Total Return: {total_return:.2%}")
        print(f"Benchmark Return: {benchmark_cum_return.iloc[-1]:.2%}")
        print(f"Total Trades: {len(trades)}")
        if not trades.empty:
            win_rate = (trades['net_pnl'] > 0).mean()
            avg_return = trades['net_return'].mean()
            max_drawdown_val = self._calculate_max_drawdown(history['aum'])
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Average Return per Trade: {avg_return:.2%}")
            print(f"Max Drawdown: {max_drawdown_val:.2%}")
        
        print(f"Entry Signals: {len(entry_signals)}")
        print(f"Exit Signals: {len(exit_signals)}")


class PerformanceAnalyzer:
    def __init__(self, results: dict, strategy_type: str):
        self.results = results
        self.strategy_type = strategy_type

    def _calculate_max_drawdown(self, aum_series: pd.Series) -> float:
        if aum_series.empty: return 0
        running_max = aum_series.cummax()
        drawdown = (aum_series - running_max) / running_max
        return drawdown.min()

    def generate_summary_table(self) -> pd.DataFrame:
        summary_data = []
        for ticker, result in self.results.items():
            history = result['history']
            trades = result['trades']
            log_df = result['log']
            
            if history.empty: continue

            total_return = history['cumulative_return'].iloc[-1]
            benchmark_return = (log_df['close'] / log_df['close'].iloc[0] - 1).iloc[-1]
            max_drawdown = self._calculate_max_drawdown(history['aum'])
            win_rate = (trades['net_pnl'] > 0).mean() if not trades.empty else 0
            
            winning_trades = trades[trades['net_pnl'] > 0]['net_pnl']
            losing_trades = trades[trades['net_pnl'] < 0]['net_pnl']
            
            avg_win = winning_trades.mean() if not winning_trades.empty else 0
            avg_loss = abs(losing_trades.mean()) if not losing_trades.empty else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf

            summary_data.append({
                'Ticker': ticker,
                'Total Return': total_return,
                'Benchmark Return': benchmark_return,
                'MDD': max_drawdown,
                'Win Rate': win_rate,
                'Profit/Loss Ratio': profit_loss_ratio,
                'Total Trades': len(trades),
                'Avg Win': avg_win,
                'Avg Loss': avg_loss,
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df.set_index('Ticker')

    def plot_ticker_analysis(self, ticker: str, vpci_threshold: float = 0.4):
        if ticker not in self.results:
            print(f"No results found for ticker {ticker}.")
            return
            
        result = self.results[ticker]
        history = result['history']
        log_df = result['log']
        trades = result['trades']
        
        if history.empty:
            print(f"No trading activity for ticker {ticker}.")
            return

        # trailing_stop_pct 추출 로직 개선
        trailing_stop_pct = 0.15  # 기본값
        if not trades.empty:
            first_exit_reason = trades.iloc[0]['exit_reason']
            if 'trailing_stop_' in first_exit_reason:
                try:
                    trailing_stop_pct = float(first_exit_reason.split('_')[-1])
                except ValueError:
                    trailing_stop_pct = 0.15
        
        # --- Benchmark Calculations ---
        benchmark_cum_return = (log_df['close'] / log_df['close'].iloc[0]) - 1
        benchmark_aum = log_df['close']
        benchmark_running_max = benchmark_aum.cummax()
        benchmark_underwater = (benchmark_aum - benchmark_running_max) / benchmark_running_max
        benchmark_max_drawdown = benchmark_underwater.min()
        
        # --- Strategy Calculations ---
        drawdown = self._calculate_max_drawdown(history['aum'])
        running_max = history['aum'].cummax()
        underwater = (history['aum'] - running_max) / running_max
        
        # --- Plotting Setup ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax_daily, ax2, ax3) = plt.subplots(4, 1, figsize=(15, 22), height_ratios=[3, 1, 1.5, 2], sharex=True)
        fig.suptitle(f'{self.strategy_type.upper()} Strategy Performance - {ticker} (Trailing Stop: {trailing_stop_pct:.1%})', fontsize=16)
        
        # --- AX1: Cumulative Return & Return Difference ---
        ax1.plot(history.index, history['cumulative_return'] * 100, label='Strategy Cumulative Return (%)', color='blue', linewidth=2)
        ax1.plot(benchmark_cum_return.index, benchmark_cum_return * 100, label='Benchmark (Buy & Hold) (%)', color='grey', linestyle='--', linewidth=1.5)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.5)

        entry_signals = log_df[log_df['signal'].isin(['vbot', 'wbot'])]
        exit_signals = log_df[log_df['signal'] == 'exit']
        
        if not entry_signals.empty:
            ax1.scatter(entry_signals.index, history.loc[entry_signals.index, 'cumulative_return'] * 100,
                        color='green', marker='^', s=100, label='Entry Signals', zorder=5)
        
        if not exit_signals.empty:
            ax1.scatter(exit_signals.index, history.loc[exit_signals.index, 'cumulative_return'] * 100,
                        color='red', marker='v', s=100, label='Exit Signals', zorder=5)
        ax1.legend(loc='upper left')

        ax1_twin = ax1.twinx()
        return_diff = (history['cumulative_return'] - benchmark_cum_return) * 100
        ax1_twin.fill_between(return_diff.index, return_diff, 0,
                              where=return_diff >= 0, color='lightgreen', alpha=0.3, interpolate=True, label='Outperformance')
        ax1_twin.fill_between(return_diff.index, return_diff, 0,
                              where=return_diff < 0, color='lightcoral', alpha=0.3, interpolate=True, label='Underperformance')
        ax1_twin.set_ylabel('Strategy vs Benchmark Return Difference (%)', color='gray')
        ax1_twin.tick_params(axis='y', labelcolor='gray')
        ax1_twin.legend(loc='upper right')
        
        # --- AX_DAILY: Daily Returns ---
        daily_returns = history['daily_return'] * 100
        ax_daily.bar(daily_returns.index, daily_returns, color=np.where(daily_returns >= 0, 'g', 'r'), alpha=0.7)
        ax_daily.set_ylabel('Daily Return (%)')
        ax_daily.set_title('Daily Returns (Strategy - Benchmark)', fontsize=12)
        ax_daily.grid(True, alpha=0.5)
        
        # --- AX2: Drawdown ---
        ax2.fill_between(underwater.index, 0, underwater * 100, color='lightblue', alpha=0.3, label=f'Strategy DD (Max: {drawdown:.2%})')
        ax2.plot(benchmark_underwater.index, benchmark_underwater * 100, color='grey', linestyle='--', label=f'Benchmark DD (Max: {benchmark_max_drawdown:.2%})')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title(f'Underwater Plot', fontsize=12)
        ax2.grid(True, alpha=0.5)
        ax2.legend(loc='lower left')
        
        # --- AX3: VPCI Plot ---
        mask = (log_df['vpci'].notna()) & (log_df['lb_vpci'].notna()) & (log_df['ub_vpci'].notna())
        clean_df = log_df[mask]
        
        ax3.plot(clean_df.index, clean_df['vpci'], label='VPCI', color='purple', linewidth=1.5)
        ax3.plot(clean_df.index, clean_df['lb_vpci'], label='LB_VPCI', color='orange', linewidth=1.5)
        ax3.plot(clean_df.index, clean_df['ub_vpci'], label='UB_VPCI', color='cyan', linewidth=1.5)
        ax3.axhline(y=vpci_threshold, color='red', linestyle='--', alpha=0.7, label=f'VPCI Threshold ({vpci_threshold:.1f})')
        
        ax3.set_ylabel('VPCI Value')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    def plot_all_tickers(self):
        if not self.results:
            print("No results to plot.")
            return
        
        print(f"\n=== Plotting {len(self.results)} tickers ===")
        for ticker in self.results.keys():
            print(f"Plotting {ticker}...")
            try:
                self.plot_ticker_analysis(ticker=ticker)
                print(f"Completed {ticker}\n")
            except Exception as e:
                print(f"Error plotting {ticker}: {e}\n")
                continue


class EventAnalyzer:
    def __init__(self, preprocess: PreProcess, strategy_type: str = 'vbot', multiplier: float = 2.0, vpci_threshold: float = 0.4):
        self.preprocess = preprocess
        self.strategy_type = strategy_type
        self.multiplier = multiplier
        self.vpci_threshold = vpci_threshold

    def _get_signals(self, close: pd.Series, volume: pd.Series) -> pd.DatetimeIndex:
        calculator = VPCICalculator(close, volume, 
                                   lower_multiplier=self.multiplier, 
                                   upper_multiplier=self.multiplier)
        df = pd.DataFrame({'close': close, 'vpci': calculator.vpci, 'lb_vpci': calculator.lb_vpci})
        
        conditions = (df['vpci'] <= self.vpci_threshold) & (df['vpci'] < df['lb_vpci'])
        
        if self.strategy_type == 'vbot':
            return df[conditions].index
        
        elif self.strategy_type == 'wbot':
            signal_dates = df[conditions].index
            wbot_signals = []
            last_vbot_date = None
            for date in signal_dates:
                if last_vbot_date is None:
                    last_vbot_date = date
                else:
                    days_since_last = (date - last_vbot_date).days
                    if 0 < days_since_last <= 20:
                        wbot_signals.append(date)
                        last_vbot_date = None
                    else:
                        last_vbot_date = date
            return pd.to_datetime(wbot_signals)
        
        return pd.DatetimeIndex([])

    def run_event_study(self, tickers: list[str], event_window: tuple[int, int], start_date: str = None, end_date: str = None):
        events_by_date = {}
        all_event_series = {}
        
        start_offset, end_offset = event_window

        for ticker in tqdm(tickers, desc=f"Analyzing {self.strategy_type.upper()} Events"):
            data = self.preprocess()
            close = data['close'][ticker].dropna()
            volume = data['volume'].get(f"{ticker}.1")
            
            if volume is None or len(close) < 52: continue

            common_index = close.index.intersection(volume.index)
            close = close[common_index]
            
            signal_dates = self._get_signals(close, volume)
            
            if start_date: signal_dates = signal_dates[signal_dates >= pd.to_datetime(start_date)]
            if end_date: signal_dates = signal_dates[signal_dates <= pd.to_datetime(end_date)]

            for signal_date in signal_dates:
                entry_price = close.get(signal_date)
                if entry_price is None or entry_price == 0: continue
                
                signal_idx_loc = close.index.get_loc(signal_date)
                
                start_idx = max(0, signal_idx_loc + start_offset)
                end_idx = min(len(close) - 1, signal_idx_loc + end_offset)
                
                event_series = close.iloc[start_idx : end_idx + 1]
                
                normalized_returns = (event_series / entry_price) - 1
                relative_days = range(start_idx - signal_idx_loc, end_idx - signal_idx_loc + 1)
                
                events_by_date[signal_date.strftime('%Y-%m-%d')] = normalized_returns
                
                event_key = f"{ticker}_{signal_date.strftime('%Y-%m-%d')}"
                all_event_series[event_key] = pd.Series(normalized_returns.values, index=relative_days)
                
        if not all_event_series:
            return {}
            
        all_returns_wide_df = pd.DataFrame(all_event_series)
        
        final_index = pd.Index(range(start_offset, end_offset + 1), name='relative_days')
        all_returns_wide_df = all_returns_wide_df.reindex(final_index)
        
        average_returns_df = pd.DataFrame(index=all_returns_wide_df.index)
        average_returns_df['mean_return'] = all_returns_wide_df.mean(axis=1)
        average_returns_df['event_count'] = all_returns_wide_df.count(axis=1)

        return {
            'average_returns': average_returns_df,
            'all_event_returns': all_returns_wide_df,
            'events_by_date': events_by_date
        }

    def generate_summary_and_plot(self, event_study_results: dict):
        if not event_study_results or 'average_returns' not in event_study_results:
            print("No event data to summarize or plot.")
            return

        summary = event_study_results['average_returns']
        
        print("\n--- Event Study Summary (Avg Return by Day) ---")
        print(summary.to_string(formatters={'mean_return': '{:.2%}'.format}))
        
        post_event_returns = summary[summary.index > 0]['mean_return']
        if not post_event_returns.empty:
            max_return = post_event_returns.max()
            max_return_day = post_event_returns.idxmax()
            min_return = post_event_returns.min()
            min_return_day = post_event_returns.idxmin()
            
            print("\n--- Post-Event Analysis ---")
            print(f"Max Average Return: {max_return:.2%} (on day {max_return_day})")
            print(f"Min Average Return: {min_return:.2%} (on day {min_return_day})")

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.plot(summary.index, summary['mean_return'] * 100, label='Average Cumulative Return', color='blue', linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Signal Event (Day 0)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        if not post_event_returns.empty:
            ax.scatter(max_return_day, max_return * 100, color='green', marker='^', s=150, zorder=5, label=f'Max Return ({max_return:.2%})')
            ax.scatter(min_return_day, min_return * 100, color='red', marker='v', s=150, zorder=5, label=f'Min Return ({min_return:.2%})')

        ax.set_title(f'Event Study: Average Return for {self.strategy_type.upper()} Strategy', fontsize=16)
        ax.set_xlabel('Days Relative to Signal Event')
        ax.set_ylabel('Average Return P_t / P_0 (%)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.show()


if __name__ == "__main__":
    preprocess = PreProcess(file_name='data_vpci')
    
    tickers_to_analyze = ['IKS200']
    event_window_to_test = (-60, 90)
    trailing_stop_pct = 0.10
    multiplier = 1.25
    vpci_threshold = 0

    # VBOT
    bt = VPCIBacktest(preprocess, ticker=tickers_to_analyze, multiplier=2)
    results, performance_df = bt.run(initial_aum=1e8, 
                                     commission_rate=0.0, 
                                     trailing_stop_pct=trailing_stop_pct, 
                                     start_date='2015-01-01', 
                                     end_date='2025-07-27', 
                                     strategy_type='vbot', 
                                     multiplier=multiplier,
                                     vpci_threshold=vpci_threshold)
    
    perf = PerformanceAnalyzer(results, strategy_type='vbot')
    summary_table = perf.generate_summary_table()
    print("=== VBOT Strategy Results ===")
    print(summary_table)
    perf.plot_ticker_analysis(ticker='IKS200', vpci_threshold=vpci_threshold)
    
    # Band
    print("\n=== Testing Band Strategy ===")
    bt_band = VPCIBacktest(preprocess, ticker=tickers_to_analyze)
    results_band, performance_df_band = bt_band.run(
        initial_aum=1e8, 
        commission_rate=0.0, 
        trailing_stop_pct=0.0,
        start_date='2015-01-01', 
        end_date='2025-07-27', 
        strategy_type='band',
        lower_multiplier=3.0,
        upper_multiplier=3.0,
        lower_threshold=0.4,
        upper_threshold=-0.8,
        use_trailing_stop=False
    )
    
    perf_band = PerformanceAnalyzer(results_band, strategy_type='band')
    summary_table_band = perf_band.generate_summary_table()
    print("=== Band Strategy Results ===")
    print(summary_table_band)
    perf_band.plot_ticker_analysis(ticker='IKS200', vpci_threshold=0.4)
    
    # Event Study
    for strategy in ['vbot', 'wbot']:
        print(f"\n\n--- Analyzing {strategy.upper()} Strategy ---")
        analyzer = EventAnalyzer(preprocess, 
                                 strategy_type=strategy, 
                                 multiplier=multiplier,
                                 vpci_threshold=vpci_threshold)
        
        event_study_results = analyzer.run_event_study(
            tickers=tickers_to_analyze,
            event_window=event_window_to_test,
            start_date='2015-01-01',
            end_date='2025-07-27'
        )
        analyzer.generate_summary_and_plot(event_study_results)