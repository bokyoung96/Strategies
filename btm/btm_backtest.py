import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

class Loader:
    def __init__(self, 
                 asset_code: str = "A233740", 
                 data_folder: str = "DATA"):
        self.source_path = os.path.join(os.path.dirname(__file__), data_folder)
        self.asset_code = asset_code
        if not os.path.isdir(self.source_path):
            raise FileNotFoundError(f"Directory not found: {self.source_path}")

    def __getattr__(self, name: str) -> pd.DataFrame:
        file_path = os.path.join(self.source_path, f"{self.asset_code}_{name}.parquet")
        if not os.path.exists(file_path):
            raise AttributeError(f"Data for '{name}' not found in '{self.source_path}'")
        try:
            return pd.read_parquet(file_path).reset_index(drop=True)
        except Exception as e:
            raise IOError(f"Failed to load {file_path}: {e}") from e

    def __dir__(self) -> list[str]:
        default_attrs = super().__dir__()
        search_path = os.path.join(self.source_path, f"{self.asset_code}_*.parquet")
        parquet_files = glob.glob(search_path)
        
        stem_pattern = f"{self.asset_code}_"
        discovered_types = []
        for f_path in parquet_files:
            stem = os.path.basename(os.path.splitext(f_path)[0])
            if stem.startswith(stem_pattern):
                discovered_types.append(stem.replace(stem_pattern, "", 1))
        return sorted(list(set(default_attrs + discovered_types)))

class FeatureEngineer:
    def __init__(self, 
                 loader: Loader, 
                 start_date: str, 
                 end_date: str, 
                 rolling_move: int,
                 atr_period: int = 14,
                 historical_atr_days: int = 20):
        self.loader = loader
        self.start_date = start_date
        self.end_date = end_date
        self.rolling_move = rolling_move
        self.atr_period = atr_period
        self.historical_atr_days = historical_atr_days

    def create_features(self) -> pd.DataFrame:
        df = self.loader.intra.copy()
        if self.start_date: df = df[df['caldt'] >= pd.to_datetime(self.start_date)]
        if self.end_date: df = df[df['caldt'] <= pd.to_datetime(self.end_date)]
        if df.empty: return df
        
        df.sort_values(by=['caldt'], inplace=True)
        df['day'] = pd.to_datetime(df['caldt']).dt.date
        df.set_index('caldt', inplace=True)
        
        # Minute-level ATR Calculation
        prev_close = df['close'].shift(1)
        
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df['atr'] = true_range.ewm(span=self.atr_period, adjust=False).mean()
        
        df['vwap'] = df.groupby('day', group_keys=False).apply(self._calculate_vwap, include_groups=False)
        df['move_open'] = df.groupby('day')['close'].transform(lambda x: (x / x.iloc[0] - 1).abs())
        df['min_from_open'] = ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1)) - 540
        df['sigma_open'] = df.groupby('min_from_open')['move_open'].transform(
            lambda x: x.rolling(window=self.rolling_move, min_periods=max(self.rolling_move//2, 3)).mean().shift(1)
        )
        df['sigma_open'] = df.groupby('day')['sigma_open'].ffill().bfill()
        
        if self.historical_atr_days > 0:
            min_periods = max(self.historical_atr_days // 2, 10)
            df['historical_avg_atr'] = df.groupby('min_from_open')['atr'].transform(
                lambda x: x.rolling(window=self.historical_atr_days, min_periods=min_periods).mean().shift(1)
            )
            df['historical_avg_atr'] = df.groupby('day')['historical_avg_atr'].ffill().bfill()

        return df.dropna(subset=['sigma_open', 'atr', 'historical_avg_atr'])

    @staticmethod
    def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        return (hlc3 * df['volume']).cumsum() / df['volume'].cumsum()

class Portfolio:
    def __init__(self, 
                 initial_aum: float, 
                 commission_rate: float,
                 tax_rate: float):
        self.initial_aum = initial_aum
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.reset()

    def reset(self):
        self.aum = self.initial_aum
        self.gross_aum = self.initial_aum
        self.position = {
            'direction': 0, 
            'shares': 0, 
            'entry_price': 0, 
            'entry_time': None, 
            'ladder_step': 0, 
            'initial_shares': 0,
            'signal_type': 'none'
        }
        self.history = pd.DataFrame(columns=['aum', 'pnl', 'gross_aum', 'gross_pnl'], dtype=float)
        self.trades = []

    def record_daily_state(self, date):
        if date not in self.history.index:
            last_aum = self.history['aum'].iloc[-1] if not self.history.empty else self.initial_aum
            last_gross_aum = self.history['gross_aum'].iloc[-1] if not self.history.empty else self.initial_aum
            self.history.loc[date] = {
                'aum': float(last_aum), 
                'pnl': 0.0, 
                'gross_aum': float(last_gross_aum), 
                'gross_pnl': 0.0
            }
            
    def execute_entry(self, direction, price, timestamp, trade_amount, signal_type="unknown"):
        shares = (trade_amount / price) * direction
        self.position = {
            'direction': direction, 
            'shares': shares, 
            'entry_price': price, 
            'entry_time': timestamp, 
            'ladder_step': 0, 
            'initial_shares': shares,
            'signal_type': signal_type
        }

    def execute_exit(self, price, timestamp, action, exit_reason="unknown"):
        pos = self.position
        
        if action['type'] == 'full':
            shares_to_exit = pos['shares']
        else:
            shares_to_exit = pos['shares'] * action['size']
        
        gross_pnl = (price - pos['entry_price']) * shares_to_exit
        commission = abs(shares_to_exit) * (pos['entry_price'] + price) * self.commission_rate
        net_pnl = gross_pnl - commission
        
        gross_return = (price / pos['entry_price'] - 1) * pos['direction']
        net_return = net_pnl / (abs(pos['initial_shares']) * pos['entry_price'])
        
        self.aum += net_pnl
        self.gross_aum += gross_pnl
        
        today = pd.to_datetime(timestamp).date()
        self.history.loc[today, 'pnl'] = float(self.history.loc[today, 'pnl'] + net_pnl)
        self.history.loc[today, 'gross_pnl'] = float(self.history.loc[today, 'gross_pnl'] + gross_pnl)
        self.history.loc[today, 'aum'] = float(self.aum)
        self.history.loc[today, 'gross_aum'] = float(self.gross_aum)
        
        self.trades.append({
            'entry_time': pos['entry_time'], 
            'exit_time': timestamp, 
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'direction': pos['direction'],
            'signal_type': pos.get('signal_type', 'unknown'),
            'exit_reason': exit_reason,
            'ladder_step': pos['ladder_step'],
            'shares': abs(shares_to_exit),
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'gross_return': gross_return,
            'net_return': net_return
        })

        if action['type'] == 'full': 
            self.reset_position()
        else:
            pos['shares'] -= shares_to_exit
            pos['ladder_step'] += 1
            
    def reset_position(self):
        self.position = {
            'direction': 0, 
            'shares': 0, 
            'entry_price': 0, 
            'entry_time': None, 
            'ladder_step': 0, 
            'initial_shares': 0,
            'signal_type': 'none'
        }

class Strategy:
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self._calculate_bands()

    def _calculate_bands(self): raise NotImplementedError
    def get_entry_signal(self, row, portfolio): raise NotImplementedError
    def get_exit_action(self, row, portfolio): raise NotImplementedError

class BandVWAPStrategy(Strategy):
    def _calculate_bands(self):
        df, mult = self.data, self.params['band_multiplier']
        open_price = df.groupby('day')['open'].transform('first')
        prev_close_map = df['day'].map(df.groupby('day')['close'].last().shift(1).ffill())
        ub_base = pd.concat([open_price, prev_close_map], axis=1).max(axis=1)
        lb_base = pd.concat([open_price, prev_close_map], axis=1).min(axis=1)
        df['UB'] = ub_base * (1 + mult * df['sigma_open'])
        df['LB'] = lb_base * (1 - mult * df['sigma_open'])
        
    def get_entry_signal(self, row, portfolio):
        if portfolio.position['direction'] != 0: return 0, "none"
        use_vwap = self.params.get('use_vwap', True)
        
        if row.close > row.UB:
            if not use_vwap or row.close > row.vwap:
                return 1, "band_cross_up"
        elif row.close < row.LB:
            if not use_vwap or row.close < row.vwap:
                return -1, "band_cross_down"
        
        return 0, "none"
        
    def get_exit_action(self, row, portfolio):
        if portfolio.position['direction'] == 0: return None
        pos_dir, use_vwap = portfolio.position['direction'], self.params.get('use_vwap', True)
        is_reentry = row.close < row.UB and row.close > row.LB
        
        vwap_exit = use_vwap and ((pos_dir == 1 and row.close < row.vwap) or (pos_dir == -1 and row.close > row.vwap))
        
        if is_reentry:
            return {'type': 'full', 'reason': 'band_reentry'}
        elif vwap_exit:
            return {'type': 'full', 'reason': 'vwap_exit'}
        
        return None

class LadderExitStrategy(BandVWAPStrategy):
    def _calculate_bands(self):
        super()._calculate_bands()
        df, entry_mult = self.data, self.params['band_multiplier']
        exit_mult = self.params.get('exit_band_multiplier', entry_mult)
        if exit_mult != entry_mult:
            open_price = df.groupby('day')['open'].transform('first')
            prev_close_map = df['day'].map(df.groupby('day')['close'].last().shift(1).ffill())
            ub_base = pd.concat([open_price, prev_close_map], axis=1).max(axis=1)
            lb_base = pd.concat([open_price, prev_close_map], axis=1).min(axis=1)
            df['exit_UB'] = ub_base * (1 + exit_mult * df['sigma_open'])
            df['exit_LB'] = lb_base * (1 - exit_mult * df['sigma_open'])
        else:
            df['exit_UB'], df['exit_LB'] = df['UB'], df['LB']

    def get_entry_signal(self, row, portfolio):
        return super().get_entry_signal(row, portfolio)

    def get_exit_action(self, row, portfolio):
        if portfolio.position['direction'] == 0: return None
        pos = portfolio.position
        ladder_levels = self.params.get('ladder_levels')
        
        # Ladder exit
        if ladder_levels and pos['ladder_step'] < len(ladder_levels):
            level_conf = ladder_levels[pos['ladder_step']]
            stop_price = pos['entry_price'] * (1 + level_conf['stop'] * pos['direction'])
            profit_price = pos['entry_price'] * (1 + level_conf['profit'] * pos['direction'])
            
            # Stop loss
            if (pos['direction'] == 1 and row.close <= stop_price) or (pos['direction'] == -1 and row.close >= stop_price):
                return {'type': 'full', 'reason': 'ladder_stop', 'level': pos['ladder_step']}
            
            # Profit taking
            if (pos['direction'] == 1 and row.close >= profit_price) or (pos['direction'] == -1 and row.close <= profit_price):
                is_last_step = pos['ladder_step'] == len(ladder_levels) - 1
                exit_type = 'full' if is_last_step else 'partial'
                return {
                    'type': exit_type, 
                    'size': level_conf['size'],
                    'reason': 'ladder_profit',
                    'level': pos['ladder_step']
                }
        
        is_reentry = row.close < row.exit_UB and row.close > row.exit_LB
        use_vwap = self.params.get('use_vwap', True)
        pos_dir = portfolio.position['direction']
        vwap_exit = use_vwap and ((pos_dir == 1 and row.close < row.vwap) or (pos_dir == -1 and row.close > row.vwap))
        
        if is_reentry:
            return {'type': 'full', 'reason': 'band_reentry'}
        elif vwap_exit:
            return {'type': 'full', 'reason': 'vwap_exit'}
        
        return None

class LadderExitMAStrategy(LadderExitStrategy):
    def __init__(self, data, params):
        self.ma_period = params.get('ma_period', 120)
        super().__init__(data, params)

    def _calculate_bands(self):
        super()._calculate_bands()
        
        daily_close = self.data.groupby('day')['close'].last()
        daily_close.index = pd.to_datetime(daily_close.index)
        
        daily_ma = daily_close.rolling(window=self.ma_period, min_periods=max(self.ma_period//2, 10)).mean()
        
        prev_daily_close = daily_close.shift(1)
        prev_daily_ma = daily_ma.shift(1)
        
        self.data['prev_daily_close'] = self.data['day'].map(prev_daily_close)
        self.data['prev_daily_ma'] = self.data['day'].map(prev_daily_ma)
        
    def get_entry_signal(self, row, portfolio):
        if portfolio.position['direction'] != 0: 
            return 0, "none"
            
        if pd.isna(row.prev_daily_ma) or pd.isna(row.prev_daily_close):
            return 0, "none"
            
        base_signal, base_signal_type = super().get_entry_signal(row, portfolio)
        
        if base_signal == 0:
            return 0, "none"
            
        above_ma = row.prev_daily_close > row.prev_daily_ma
        below_ma = row.prev_daily_close < row.prev_daily_ma
        
        if base_signal == 1 and above_ma:
            return 1, f"{base_signal_type}_ma_bull"
        elif base_signal == -1 and below_ma:
            return -1, f"{base_signal_type}_ma_bear"
        else:
            return 0, "ma_filtered"

    def get_exit_action(self, row, portfolio):
        return super().get_exit_action(row, portfolio)

class ATRStopStrategy(BandVWAPStrategy):
    def get_exit_action(self, row, portfolio):
        if portfolio.position['direction'] == 0: return None
        
        pos = portfolio.position
        direction = pos['direction']
        entry_price = pos['entry_price']
        
        atr_multiplier_stop = self.params.get('atr_multiplier_stop', 2.0)
        atr_multiplier_profit = self.params.get('atr_multiplier_profit', 4.0)

        stop_loss_price = entry_price - direction * row.atr * atr_multiplier_stop
        take_profit_price = entry_price + direction * row.atr * atr_multiplier_profit
        
        if direction == 1:
            if row.close <= stop_loss_price:
                return {'type': 'full', 'reason': 'atr_stop'}
            if row.close >= take_profit_price:
                return {'type': 'full', 'reason': 'atr_profit'}
        elif direction == -1:
            if row.close >= stop_loss_price:
                return {'type': 'full', 'reason': 'atr_stop'}
            if row.close <= take_profit_price:
                return {'type': 'full', 'reason': 'atr_profit'}
        return super().get_exit_action(row, portfolio)


class Backtester:
    def __init__(self, data: pd.DataFrame, portfolio: Portfolio, strategy: Strategy, params: dict):
        self.data = data
        self.portfolio = portfolio
        self.strategy = strategy
        self.params = params

    def run(self):
        for day, day_data in tqdm(self.data.groupby(self.data.index.date), desc="Backtesting"):
            self.portfolio.record_daily_state(day)
            daily_trade_amount = self.portfolio.history.loc[day, 'aum']
            
            for i, row in enumerate(day_data.itertuples()):
                is_eod = i == len(day_data) - 1
                is_trade_time = (row.min_from_open >= 0 and row.min_from_open % self.params['trade_freq'] == 0)

                if self.portfolio.position['direction'] != 0:
                    action = self.strategy.get_exit_action(row, self.portfolio)
                    if action or is_eod:
                        if action and 'reason' in action:
                            reason = action['reason']
                            if reason == 'ladder_stop':
                                exit_reason = f"L{action['level']} Stop"
                            elif reason == 'ladder_profit':
                                level = action['level']
                                size_pct = int(action.get('size', 1.0) * 100)
                                exit_reason = f"L{level} Profit ({size_pct}%)"
                            elif reason == 'band_reentry':
                                exit_reason = "Band Reentry"
                            elif reason == 'vwap_exit':
                                exit_reason = "VWAP Exit"
                            elif reason == 'atr_stop':
                                exit_reason = "ATR Stop"
                            elif reason == 'atr_profit':
                                exit_reason = "ATR Profit"
                            else:
                                exit_reason = reason
                        elif is_eod:
                            exit_reason = "EOD"
                        else:
                            exit_reason = "Unknown"
                        
                        self.portfolio.execute_exit(row.close, row.Index, action or {'type': 'full'}, exit_reason)
                        if is_eod: continue
                
                if is_trade_time:
                    signal, signal_type = self.strategy.get_entry_signal(row, self.portfolio)
                    if signal != 0:
                        self.portfolio.execute_entry(signal, row.close, row.Index, daily_trade_amount, signal_type)
        
        history = self.portfolio.history.copy()
        history['daily_return'] = history['aum'].pct_change().fillna(0)
        history['daily_gross_return'] = history['gross_aum'].pct_change().fillna(0)
        history['cumulative_return'] = (history['aum'] / self.portfolio.initial_aum) - 1
        history['cumulative_gross_return'] = (history['gross_aum'] / self.portfolio.initial_aum) - 1
        return history, pd.DataFrame(self.portfolio.trades)

class Reporter:
    def __init__(self, daily_results, trades, initial_aum, feature_data=None):
        self.daily_results = daily_results
        self.trades = trades
        self.initial_aum = initial_aum
        self.feature_data = feature_data
        self.strategy_params = {}

    def _calculate_drawdowns(self):
        if self.daily_results.empty:
            return None, None, None, None
            
        cumulative_returns = self.daily_results['aum'] / self.initial_aum
        running_max = cumulative_returns.cummax()
        
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        gross_cumulative_returns = self.daily_results['gross_aum'] / self.initial_aum
        gross_running_max = gross_cumulative_returns.cummax()
        gross_drawdowns = (gross_cumulative_returns - gross_running_max) / gross_running_max
        gross_max_drawdown = gross_drawdowns.min()
        
        return drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown

    def print_summary(self):
        if self.daily_results.empty: print("No results to summarize."); return
        
        # Calculate total returns
        total_net_return = (self.daily_results['aum'].iloc[-1] / self.initial_aum) - 1
        total_gross_return = (self.daily_results['gross_aum'].iloc[-1] / self.initial_aum) - 1
        total_trades = len(self.trades)
        
        # Calculate CAGR
        start_date = self.daily_results.index[0]
        end_date = self.daily_results.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25 if days > 0 else 1
        
        net_cagr = ((self.daily_results['aum'].iloc[-1] / self.initial_aum) ** (1/years) - 1) if years > 0 else 0
        gross_cagr = ((self.daily_results['gross_aum'].iloc[-1] / self.initial_aum) ** (1/years) - 1) if years > 0 else 0
        
        # Calculate maximum drawdowns
        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        
        summary = {
            "Total Net Return": f"{total_net_return:.2%}",
            "Total Gross Return": f"{total_gross_return:.2%}",
            "CAGR (Net)": f"{net_cagr:.2%}",
            "CAGR (Gross)": f"{gross_cagr:.2%}",
            "Maximum Drawdown (Net)": f"{max_drawdown:.2%}" if max_drawdown is not None else "N/A",
            "Maximum Drawdown (Gross)": f"{gross_max_drawdown:.2%}" if gross_max_drawdown is not None else "N/A",
            "Total Trades": total_trades,
        }
        
        if not self.trades.empty:
            # Calculate win rate and average returns
            win_rate = (self.trades['net_pnl'] > 0).mean()
            avg_gross_return = self.trades['gross_return'].mean()
            avg_net_return = self.trades['net_return'].mean()
            
            # Calculate commission metrics
            total_commission = self.trades['commission'].sum()
            avg_commission_per_trade = total_commission / total_trades if total_trades > 0 else 0
            
            # Calculate profit/loss ratio
            winning_trades = self.trades[self.trades['net_return'] > 0]['net_return']
            losing_trades = self.trades[self.trades['net_return'] < 0]['net_return']
            
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                avg_win = winning_trades.mean()
                avg_loss = abs(losing_trades.mean())
                profit_loss_ratio = avg_win / avg_loss
            else:
                profit_loss_ratio = None
        
            signal_breakdown = self.trades['signal_type'].value_counts()
            
            summary.update({
                "Win Rate": f"{win_rate:.2%}",
                "Avg Gross Return/Trade": f"{avg_gross_return:.2%}",
                "Avg Net Return/Trade": f"{avg_net_return:.2%}",
                "Profit/Loss Ratio": f"{profit_loss_ratio:.2f}" if profit_loss_ratio is not None else "N/A",
                "Total Commission": f"{total_commission:,.0f}",
                "Avg Commission/Trade": f"{avg_commission_per_trade:,.0f}",
            })
        else:
            summary["Win Rate"] = "N/A"
        
        print("\n--- Backtest Summary ---")
        [print(f"{k:<25}: {v}") for k, v in summary.items()]
        
        if not self.trades.empty and len(signal_breakdown) > 0:
            print(f"\n--- Signal Type Breakdown ---")
            for signal_type, count in signal_breakdown.items():
                pct = count / total_trades * 100
                avg_return = self.trades[self.trades['signal_type'] == signal_type]['net_return'].mean()
                print(f"{signal_type:<20}: {count:>3} trades ({pct:>5.1f}%) | Avg Return: {avg_return:>6.2%}")
                
        print("-" * 50)

    def plot_aum_curve(self):
        if self.daily_results.empty: return
        
        self.daily_results.index = pd.to_datetime(self.daily_results.index)
        
        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 18), constrained_layout=True)
        
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.5], hspace=0.2)
        gs_top = gs[0].subgridspec(3, 1, hspace=0)
        
        ax1 = fig.add_subplot(gs_top[0])
        ax2 = fig.add_subplot(gs_top[1], sharex=ax1)
        ax3 = fig.add_subplot(gs_top[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[1])

        ax1.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='x', labelbottom=False)
        
        self.daily_results['gross_aum'].plot(ax=ax1, label='AUM (Gross, Pre-Cost)', color='green', linestyle='--')
        self.daily_results['aum'].plot(ax=ax1, label='AUM (Net, Post-Cost)', color='blue')
        
        if not self.daily_results.empty and not self.daily_results['aum'].empty:
            start_date, end_date = self.daily_results.index[0], self.daily_results.index[-1]
            total_days = (end_date - start_date).days
            if total_days <= 0:
                total_days = 1

            points_to_annotate = {}
            
            peak_aum_date = self.daily_results['aum'].idxmax()
            points_to_annotate['peak_aum'] = {
                'date': peak_aum_date,
                'value': self.daily_results.loc[peak_aum_date, 'aum'],
                'text': f'Peak AUM\n{self.daily_results.loc[peak_aum_date, "aum"]:,.0f}\n{peak_aum_date.strftime("%Y-%m-%d")}'
            }
            trough_aum_date = self.daily_results['aum'].idxmin()
            points_to_annotate['trough_aum'] = {
                'date': trough_aum_date,
                'value': self.daily_results.loc[trough_aum_date, 'aum'],
                'text': f'Trough AUM\n{self.daily_results.loc[trough_aum_date, "aum"]:,.0f}\n{trough_aum_date.strftime("%Y-%m-%d")}'
            }

            if 'daily_return' in self.daily_results.columns:
                best_return_date = self.daily_results['daily_return'].idxmax()
                best_return_val = self.daily_results.loc[best_return_date, 'daily_return']
                points_to_annotate['best_return'] = {
                    'date': best_return_date,
                    'value': self.daily_results.loc[best_return_date, 'aum'],
                    'text': f'Best Return\n{best_return_val:+.2%}\n{best_return_date.strftime("%Y-%m-%d")}'
                }
                worst_return_date = self.daily_results['daily_return'].idxmin()
                worst_return_val = self.daily_results.loc[worst_return_date, 'daily_return']
                points_to_annotate['worst_return'] = {
                    'date': worst_return_date,
                    'value': self.daily_results.loc[worst_return_date, 'aum'],
                    'text': f'Worst Return\n{worst_return_val:+.2%}\n{worst_return_date.strftime("%Y-%m-%d")}'
                }
            
            annotation_config = {
                'peak_aum':     {'va': 'bottom', 'offset_y': 15,   'color': 'green',  'marker': 'o'},
                'trough_aum':   {'va': 'top',    'offset_y': -15,  'color': 'red',    'marker': 'o'},
                'best_return':  {'va': 'bottom', 'offset_y': 40,   'color': 'blue',   'marker': '*'},
                'worst_return': {'va': 'top',    'offset_y': -40,  'color': 'purple', 'marker': '*'},
            }

            for key, point in points_to_annotate.items():
                config = annotation_config.get(key)
                if not config: continue
                
                is_left_half = (point['date'] - start_date).days < total_days / 2
                ha = 'left' if is_left_half else 'right'
                xytext_x = 50 if is_left_half else -50
                connection_rad = 0.2 if is_left_half else -0.2
                
                plot_date = point['date'].to_pydatetime()

                ax1.scatter(plot_date, point['value'], marker=config['marker'], s=150, facecolors='none', edgecolors=config['color'], linewidths=2.5, zorder=5)
                ax1.annotate(point['text'],
                             xy=(plot_date, point['value']),
                             xytext=(xytext_x, config['offset_y']),
                             textcoords='offset points',
                             ha=ha,
                             va=config['va'],
                             arrowprops=dict(arrowstyle='-|>,' + f'head_width=0.4,' + f'head_length=0.8',
                                             connectionstyle=f'arc3,rad={connection_rad}',
                                             color=config['color'],
                                             lw=1.5),
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=config['color'], lw=1, alpha=0.8))
        
        price_drawdowns = None
        max_price_dd = None
        
        if self.feature_data is not None and not self.feature_data.empty:
            daily_close = self.feature_data.groupby(self.feature_data.index.date)['close'].last()
            
            daily_results_datetime_index = self.daily_results.index
            daily_close = daily_close[daily_close.index.isin(daily_results_datetime_index.date)]
            
            if not daily_close.empty:
                ax1_twin = ax1.twinx()
                
                initial_price = daily_close.iloc[0]
                close_cumulative_return = (daily_close / initial_price - 1) * 100
                close_cumulative_return.plot(ax=ax1_twin, label='Buy&Hold Return (%)', color='gray', alpha=0.8, linewidth=1.5)
                ax1_twin.set_ylabel('Buy&Hold Cumulative Return (%)', color='gray')
                ax1_twin.tick_params(axis='y', labelcolor='gray')
                
                price_returns = daily_close / initial_price
                price_running_max = price_returns.cummax()
                price_drawdowns = (price_returns - price_running_max)
                max_price_dd = price_drawdowns.min()
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax1_twin.legend().set_visible(False)
        
        ax1.set_title('AUM and Buy&Hold Cumulative Return Over Time', fontsize=16)
        ax1.set_ylabel('AUM')
        if self.feature_data is None or self.feature_data.empty:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if drawdowns is not None:
            gross_drawdowns.plot(ax=ax2, label=f'Gross DD (Max: {gross_max_drawdown:.2%})', color='red', linestyle='--', alpha=0.7)
            drawdowns.plot(ax=ax2, label=f'Net DD (Max: {max_drawdown:.2%})', color='darkred')
            ax2.fill_between(drawdowns.index, drawdowns, 0, alpha=0.3, color='red')
            
            if price_drawdowns is not None:
                ax2_twin = ax2.twinx()
                price_drawdowns.plot(ax=ax2_twin, label=f'Buy&Hold DD (Max: {max_price_dd:.2%})', color='gray', alpha=0.8, linewidth=1.5)
                ax2_twin.set_ylabel('Buy&Hold Drawdown (%)', color='gray')
                ax2_twin.tick_params(axis='y', labelcolor='gray')
                ax2_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
                ax2_twin.legend().set_visible(False)
            
            ax2.set_title('Strategy vs Buy&Hold Drawdown Over Time', fontsize=14)
            ax2.set_ylabel('Strategy Drawdown (%)')
            if price_drawdowns is None:
                ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        if self.feature_data is not None and not self.feature_data.empty and 'open' in self.feature_data.columns:
            daily_agg = self.feature_data.groupby(self.feature_data.index.date).agg(
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last')
            )
            daily_agg.index = pd.to_datetime(daily_agg.index)
            
            intraday_power = ((daily_agg['close'] - daily_agg['open']).abs() / 
                              (daily_agg['high'] - daily_agg['low']).abs())
            intraday_power.replace([np.inf, -np.inf], np.nan, inplace=True)
            intraday_power.fillna(0, inplace=True)
            
            daily_results_datetime_index = self.daily_results.index
            intraday_power = intraday_power[intraday_power.index.isin(daily_results_datetime_index)]
            
            if not intraday_power.empty:
                ax3.bar(intraday_power.index, intraday_power, color='purple', alpha=0.6, width=1.0)
                
                ax3.set_title('Intraday Power Ratio (|Close-Open| / |High-Low|)', fontsize=14)
                ax3.set_ylabel('Ratio')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(bottom=0)
                ax3.set_xlabel('Date')
                ax3.tick_params(axis='x', rotation=45)

                daily_returns = self.daily_results['daily_return'].copy()
                combined_data = pd.DataFrame({'power_ratio': intraday_power, 'daily_return': daily_returns}).dropna()
                combined_data.index = pd.to_datetime(combined_data.index)

                if len(combined_data) >= 5:
                    try:
                        labels = ['Strong Loss', 'Loss', 'Neutral', 'Profit', 'Strong Profit']
                        
                        ranks = combined_data['daily_return'].rank(method='first')
                        combined_data['return_category'] = pd.qcut(ranks, q=5, labels=labels)
                        
                        gb = combined_data.groupby('return_category', observed=True)['daily_return']
                        avg_power_by_return = combined_data.groupby('return_category', observed=True)['power_ratio'].mean()
                        
                        if isinstance(avg_power_by_return.index, pd.CategoricalDtype):
                            avg_power_by_return = avg_power_by_return.reindex(labels)

                        colors = plt.get_cmap('gray_r')(np.linspace(0.1, 0.7, len(labels)))
                        avg_power_by_return.plot(kind='bar', ax=ax4, color=colors, alpha=0.8)
                        
                        bins_min = gb.min()
                        bins_max = gb.max()
                        bin_labels = [f"{bins_min[label]:.2%} to {bins_max[label]:.2%}" for label in avg_power_by_return.index if label in bins_min]

                        for i, patch in enumerate(ax4.patches):
                            if i < len(bin_labels):
                                ax4.text(patch.get_x() + patch.get_width() / 2.,
                                         patch.get_height(),
                                         bin_labels[i],
                                         ha='center',
                                         va='bottom',
                                         fontsize=9,
                                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.6, edgecolor='none'))
                        
                        ax4.set_title('Average Power Ratio by Daily Return Quintile', fontsize=14)
                        ax4.set_ylabel('Average Ratio')
                        ax4.set_xlabel('Return Category')
                        ax4.tick_params(axis='x', rotation=0)
                        ax4.grid(True, alpha=0.3, axis='y')

                    except Exception as e:
                        print(f"An unexpected error occurred while creating return quantiles: {e}")
                else:
                    print("Not enough daily returns to create 5 quantiles for analysis.")

        plt.show()

    def plot_candlestick_performance(self):
        if self.trades.empty or self.feature_data is None:
            print("Not enough data for candlestick performance analysis.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        daily_ohlc = self.feature_data.groupby(self.feature_data.index.date).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last')
        )
        daily_ohlc.index = pd.to_datetime(daily_ohlc.index)

        analyzer = CandlestickAnalyzer()
        daily_ohlc['pattern'] = analyzer.classify(daily_ohlc)
        
        trades_df = self.trades.copy()
        trades_df['entry_day'] = pd.to_datetime(trades_df['entry_time']).dt.normalize()
        
        merged_df = pd.merge(trades_df, daily_ohlc[['pattern']], left_on='entry_day', right_index=True)
        
        if merged_df.empty:
            print("No trades could be matched with candlestick patterns.")
            return
            
        performance = merged_df.groupby('pattern')['net_return'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        if performance.empty:
            print("No performance data to plot for candlestick patterns.")
            return

        korean_patterns = {
            "Doji": "십자형",
            "Bullish Marubozu": "장대양봉",
            "Bearish Marubozu": "장대음봉",
            "Hammer / Hanging Man": "망치형/교수형",
            "Shooting Star": "유성형",
            "Spinning Top": "팽이형",
            "Standard Candle": "보통형"
        }
        performance.rename(index=korean_patterns, inplace=True)

        fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
        
        colors = ['green' if x > 0 else 'red' for x in performance['mean']]
        
        bars = sns.barplot(x=performance.index, y=performance['mean'], ax=ax, palette=colors)
        
        for i, bar in enumerate(bars.patches):
            mean_return = performance['mean'].iloc[i]
            trade_count = performance['count'].iloc[i]
            label = f"{mean_return:.2%}\n({trade_count} trades)"
            
            y_pos = bar.get_height()
            va = 'bottom' if mean_return >= 0 else 'top'
            offset = 0.0005 if mean_return >=0 else -0.0005
            
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset, label,
                    ha='center', va=va,
                    fontsize=10, color='black')
                    
        ax.set_title('Average Net Return by Candlestick Pattern on Entry Day', fontsize=16)
        ax.set_ylabel('Average Net Return (%)')
        ax.set_xlabel('캔들 패턴')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.show()

    def plot_drawdown_analysis(self):
        if self.daily_results.empty:
            print("No data available for drawdown analysis.")
            return
            
        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        if drawdowns is None:
            print("Unable to calculate drawdowns.")
            return
            
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        drawdowns.plot(ax=ax1, color='red', alpha=0.8)
        ax1.fill_between(drawdowns.index, drawdowns, 0, alpha=0.3, color='red')
        ax1.set_title(f'Net Drawdown (Max: {max_drawdown:.2%})')
        ax1.set_ylabel('Drawdown')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax1.grid(True, alpha=0.3)
        
        drawdowns.hist(bins=50, ax=ax2, alpha=0.7, color='red')
        ax2.axvline(max_drawdown, color='darkred', linestyle='--', linewidth=2, label=f'Max DD: {max_drawdown:.2%}')
        ax2.set_title('Drawdown Distribution')
        ax2.set_xlabel('Drawdown')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        cumulative_returns = self.daily_results['aum'] / self.initial_aum
        running_max = cumulative_returns.cummax()
        
        (cumulative_returns - 1).plot(ax=ax3, label='Cumulative Return', color='blue')
        (running_max - 1).plot(ax=ax3, label='Running Peak', color='green', linestyle='--')
        ax3.fill_between(cumulative_returns.index, cumulative_returns - 1, running_max - 1, alpha=0.3, color='red')
        ax3.set_title('Underwater Plot')
        ax3.set_ylabel('Return')
        ax3.legend()
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax3.grid(True, alpha=0.3)
        
        rolling_dd = drawdowns.rolling(window=30).min()
        rolling_dd.plot(ax=ax4, color='orange', label='30-Day Rolling Max DD')
        ax4.set_title('Rolling Maximum Drawdown (30 Days)')
        ax4.set_ylabel('Max Drawdown')
        ax4.set_xlabel('Date')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.suptitle(f'Drawdown Analysis - MDD: {max_drawdown:.2%}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

    def plot_intraday(self, target_date: str):
        if self.feature_data is None:
            print("Feature data not available. Please pass feature_data to Reporter.")
            return
            
        try:
            target_date_obj = pd.to_datetime(target_date).date()
        except:
            print(f"Invalid date format: {target_date}. Please use 'YYYY-MM-DD' format.")
            return

        available_days = sorted(self.feature_data['day'].unique())
        
        if target_date_obj not in available_days:
            print(f"Date {target_date} is not a trading day or no data available.")
            before_dates = [d for d in available_days if d < target_date_obj][-5:]
            after_dates = [d for d in available_days if d > target_date_obj][:5]
            print("Nearby trading days:")
            if before_dates: print(f"  Before: {', '.join([str(d) for d in before_dates])}")
            if after_dates: print(f"  After: {', '.join([str(d) for d in after_dates])}")
            return

        day_data = self.feature_data[self.feature_data['day'] == target_date_obj].copy()
        if day_data.empty:
            print(f"No data available for date {target_date}.")
            return

        day_trades = []
        if not self.trades.empty:
            for _, trade in self.trades.iterrows():
                entry_date = pd.to_datetime(trade['entry_time']).date()
                exit_date = pd.to_datetime(trade['exit_time']).date()
                if entry_date == target_date_obj or exit_date == target_date_obj:
                    day_trades.append(trade)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        time_labels = day_data.index.strftime('%H:%M')
        x_pos = range(len(day_data))

        ax.plot(x_pos, day_data['close'], label='Close Price', color='black', linewidth=2.5, zorder=3)
        
        if 'vwap' in day_data.columns and day_data['vwap'].notna().any():
            ax.plot(x_pos, day_data['vwap'], label='VWAP', color='purple', linewidth=2, alpha=0.8, zorder=2)
        
        if 'UB' in day_data.columns and day_data['UB'].notna().any():
            ax.plot(x_pos, day_data['UB'], label='Entry Upper Band', color='red', linestyle='-', alpha=0.8, linewidth=1.5)
            ax.plot(x_pos, day_data['LB'], label='Entry Lower Band', color='blue', linestyle='-', alpha=0.8, linewidth=1.5)
            ax.fill_between(x_pos, day_data['UB'], day_data['LB'], alpha=0.08, color='gray', label='Entry Band Zone')

        if 'exit_UB' in day_data.columns and day_data['exit_UB'].notna().any():
            ax.plot(x_pos, day_data['exit_UB'], label='Exit Upper Band', color='darkred', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.plot(x_pos, day_data['exit_LB'], label='Exit Lower Band', color='darkblue', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.fill_between(x_pos, day_data['exit_UB'], day_data['exit_LB'], alpha=0.05, color='orange', label='Exit Band Zone')

        trade_events = []
        trade_colors = ['green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
        
        day_trades_sorted = sorted(day_trades, key=lambda x: pd.to_datetime(x['entry_time']))
        
        for trade_idx, trade in enumerate(day_trades_sorted):
            color_idx = trade_idx % len(trade_colors)
            base_color = trade_colors[color_idx]
            
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            entry_idx = None
            exit_idx = None
            
            if entry_time.date() == target_date_obj:
                entry_idx = day_data.index.get_indexer([entry_time], method='nearest')[0]
                if entry_idx >= 0:
                    direction = 'long' if trade['direction'] > 0 else 'short'
                    marker = '^' if direction == 'long' else 'v'
                    ax.scatter(entry_idx, trade['entry_price'], color=base_color, marker=marker, s=150, zorder=6, edgecolors='white', linewidth=1)
                    signal_type = trade.get('signal_type', 'unknown')
                    trade_events.append(f"T{trade_idx+1}: {entry_time.strftime('%H:%M')} - {direction.upper()} Entry ({signal_type}): {trade['entry_price']:.0f}")
            
            if exit_time.date() == target_date_obj:
                exit_idx = day_data.index.get_indexer([exit_time], method='nearest')[0]
                if exit_idx >= 0:
                    ax.scatter(exit_idx, trade['exit_price'], color=base_color, marker='x', s=150, zorder=6, linewidth=3)
                    net_return = trade.get('net_return', 0)
                    commission = trade.get('commission', 0)
                    exit_reason = trade.get('exit_reason', 'Unknown')
                    trade_events.append(f"T{trade_idx+1}: {exit_time.strftime('%H:%M')} - Exit ({exit_reason}): {trade['exit_price']:.0f} | Return: {net_return:.2%} | Cost: {commission:.0f}")

            if entry_idx is not None:
                start_x = entry_idx
                end_x = exit_idx if exit_idx is not None else len(x_pos) - 1

                # ATR Stop/Profit Lines
                if 'atr_multiplier_stop' in self.strategy_params:
                    entry_atr = day_data.iloc[entry_idx]['atr']
                    atr_stop = self.strategy_params.get('atr_multiplier_stop', 2.0)
                    atr_profit = self.strategy_params.get('atr_multiplier_profit', 4.0)
                    
                    stop_price = trade['entry_price'] - trade['direction'] * entry_atr * atr_stop
                    profit_price = trade['entry_price'] + trade['direction'] * entry_atr * atr_profit
                    
                    ax.hlines(stop_price, start_x, end_x, colors='darkred', linestyles='-.', alpha=0.9, linewidth=1.5,
                              label=f'ATR Stop ({atr_stop}x)' if trade_idx == 0 else "")
                    ax.text(start_x + 2, stop_price, f'T{trade_idx+1} ATR Stop', fontsize=8, color='darkred',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    ax.hlines(profit_price, start_x, end_x, colors='darkgreen', linestyles='-.', alpha=0.9, linewidth=1.5,
                              label=f'ATR Profit ({atr_profit}x)' if trade_idx == 0 else "")
                    ax.text(start_x + 2, profit_price, f'T{trade_idx+1} ATR Profit', fontsize=8, color='darkgreen',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

                # Ladder Stop/Profit Lines
                if 'ladder_levels' in self.strategy_params:
                    ladder_levels = self.strategy_params.get('ladder_levels', [])
                    entry_price = trade['entry_price']
                    direction = trade['direction']
                    
                    for i, level in enumerate(ladder_levels):
                        stop_price = entry_price * (1 + level['stop'] * direction)
                        profit_price = entry_price * (1 + level['profit'] * direction)
                        
                        ax.hlines(stop_price, start_x, end_x, colors='red', linestyles=':', alpha=0.8, linewidth=2, 
                                 label=f'L{i} Stop ({level["stop"]*100:+.1f}%)' if trade_idx == 0 and i == 0 else "")
                        ax.text(start_x + 2, stop_price, f'T{trade_idx+1}L{i}S', fontsize=8, color='red', 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                        
                        ax.hlines(profit_price, start_x, end_x, colors='green', linestyles=':', alpha=0.8, linewidth=2,
                                 label=f'L{i} Profit ({level["profit"]*100:+.1f}%)' if trade_idx == 0 and i == 0 else "")
                        ax.text(start_x + 2, profit_price, f'T{trade_idx+1}L{i}P({level["size"]*100:.0f}%)', fontsize=8, color='green',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        daily_pnl = None
        daily_return = None
        if target_date_obj in self.daily_results.index:
            daily_pnl = self.daily_results.loc[target_date_obj, 'pnl']
            if 'daily_return' in self.daily_results.columns:
                daily_return = self.daily_results.loc[target_date_obj, 'daily_return']

        title = f'Intraday Chart - {target_date}'
        if daily_pnl is not None:
            title += f' | Daily PnL: {daily_pnl:,.0f}'
            if daily_return is not None:
                title += f' ({daily_return:.2%})'
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        
        tick_interval = max(1, len(day_data) // 15)
        ax.set_xticks(x_pos[::tick_interval])
        ax.set_xticklabels(time_labels[::tick_interval], rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        if trade_events:
            print(f"\n=== {target_date} Trade Summary ===")
            if daily_pnl is not None:
                print(f"Daily PnL: {daily_pnl:,.0f}")
                if daily_return is not None:
                    print(f"Daily Return: {daily_return:.2%}")
            
            print("\nTrade Details:")
            for event in trade_events:
                print(f"  {event}")
                
            entry_trades = [event for event in trade_events if 'Entry' in event]
            exit_trades = [event for event in trade_events if 'Exit' in event]
            
            print(f"\n=== Daily Trading Statistics ===")
            print(f"  Entries: {len(entry_trades)} | Exits: {len(exit_trades)}")
            
            exit_reasons = {}
            for event in exit_trades:
                if '(' in event and ')' in event:
                    reason_start = event.find('Exit (') + 6
                    reason_end = event.find(')', reason_start)
                    if reason_start > 5 and reason_end > reason_start:
                        reason = event[reason_start:reason_end]
                        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            if exit_reasons:
                print(f"  Exit Reasons:")
                for reason, count in exit_reasons.items():
                    print(f"    {reason}: {count}")
            
            day_trade_returns = [trade.get('net_return', 0) for trade in day_trades]
            day_trade_commissions = [trade.get('commission', 0) for trade in day_trades]
            
            if day_trade_returns:
                print(f"\n=== Performance Metrics ===")
                print(f"  Total Completed Trades: {len(day_trades)}")
                print(f"  Winning Trades: {sum(1 for r in day_trade_returns if r > 0)}")
                print(f"  Losing Trades: {sum(1 for r in day_trade_returns if r < 0)}")
                print(f"  Win Rate: {sum(1 for r in day_trade_returns if r > 0) / len(day_trade_returns):.1%}")
                print(f"  Average Return: {np.mean(day_trade_returns):.2%}")
                print(f"  Best Trade: {max(day_trade_returns):.2%}")
                print(f"  Worst Trade: {min(day_trade_returns):.2%}")
                print(f"  Total Commission: {sum(day_trade_commissions):,.0f}")
                
            if hasattr(self, 'ladder_levels') and self.ladder_levels:
                print(f"\n=== Ladder Level Analysis ===")
                for trade_idx, trade in enumerate(day_trades_sorted):
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    
                    if entry_time.date() == target_date_obj:
                        entry_price = trade['entry_price']
                        direction = trade['direction']
                        exit_reason = trade.get('exit_reason', 'Unknown')
                        
                        trade_data = day_data[(day_data.index >= entry_time) & (day_data.index <= exit_time)]
                        if not trade_data.empty:
                            min_price = trade_data['low'].min()
                            max_price = trade_data['high'].max()
                            
                            print(f"\nT{trade_idx+1} ({entry_time.strftime('%H:%M')}-{exit_time.strftime('%H:%M')}) Exit: {exit_reason}")
                            print(f"  Entry: {entry_price:.0f}, Exit: {trade['exit_price']:.0f}")
                            print(f"  Price Range: {min_price:.0f} - {max_price:.0f}")
                            
                            for i, level in enumerate(self.ladder_levels):
                                stop_price = entry_price * (1 + level['stop'] * direction)
                                profit_price = entry_price * (1 + level['profit'] * direction)
                                
                                if direction == 1:  # Long
                                    stop_reached = min_price <= stop_price
                                    profit_reached = max_price >= profit_price
                                else:  # Short
                                    stop_reached = max_price >= stop_price
                                    profit_reached = min_price <= profit_price
                                
                                print(f"  L{i} Stop {stop_price:.0f}: {'✓' if stop_reached else '✗'}")
                                print(f"  L{i} Profit {profit_price:.0f}: {'✓' if profit_reached else '✗'}")
        else:
            print(f"\nNo trades occurred on {target_date}.")
            if daily_pnl is not None:
                print(f"Daily PnL: {daily_pnl:,.0f}")
                if daily_return is not None:
                    print(f"Daily Return: {daily_return:.2%}")


class CandlestickAnalyzer:
    def __init__(self,
                 body_thresh: float = 0.8,
                 wick_ratio_thresh: float = 2.0,
                 doji_thresh: float = 0.1,
                 spinning_top_body_thresh: float = 0.3):
        self.params = {
            'body_thresh': body_thresh,
            'wick_ratio_thresh': wick_ratio_thresh,
            'doji_thresh': doji_thresh,
            'spinning_top_body_thresh': spinning_top_body_thresh,
        }

    def classify(self, ohlc_df: pd.DataFrame) -> pd.Series:
        return ohlc_df.apply(self._get_pattern, axis=1)

    def _get_pattern(self, row: pd.Series) -> str:
        body = abs(row.close - row.open)
        total_range = row.high - row.low

        if total_range < 1e-9:
            return "Doji"

        body_ratio = body / total_range
        upper_wick = row.high - max(row.open, row.close)
        lower_wick = min(row.open, row.close) - row.low

        is_bullish = row.close > row.open

        if body_ratio < self.params['doji_thresh']:
            return "Doji"

        if body_ratio > self.params['body_thresh']:
            return "Bullish Marubozu" if is_bullish else "Bearish Marubozu"
        
        body_not_zero = body > 1e-9
        
        if body_not_zero and lower_wick / body > self.params['wick_ratio_thresh'] and upper_wick / total_range < self.params['doji_thresh']:
            return "Hammer / Hanging Man"

        if body_not_zero and upper_wick / body > self.params['wick_ratio_thresh'] and lower_wick / total_range < self.params['doji_thresh']:
            return "Shooting Star"

        if body_ratio < self.params['spinning_top_body_thresh']:
            return "Spinning Top"
        return "Standard Candle"

class RollingCandleStrategy(BandVWAPStrategy):
    def __init__(self, data, params):
        self.rolling_periods = params.get('rolling_periods', [30, 60, 120])
        self.strength_threshold = params.get('strength_threshold', 0.7)
        self.min_confirmation = params.get('min_confirmation', 2)
        super().__init__(data, params)

    def _calculate_bands(self):
        super()._calculate_bands()
        self._calculate_rolling_candles()

    def _calculate_rolling_candles(self):
        df = self.data
        for period in self.rolling_periods:
            df[f'roll_{period}_open'] = df['open'].shift(period - 1)
            df[f'roll_{period}_high'] = df['high'].rolling(window=period).max()
            df[f'roll_{period}_low'] = df['low'].rolling(window=period).min()
            df[f'roll_{period}_close'] = df['close']

            roll_range = df[f'roll_{period}_high'] - df[f'roll_{period}_low']
            roll_body = df[f'roll_{period}_close'] - df[f'roll_{period}_open']
            
            df[f'tsi_{period}'] = (roll_body / roll_range).replace([np.inf, -np.inf], 0).fillna(0)

    def get_entry_signal(self, row, portfolio):
        base_signal, base_signal_type = super().get_entry_signal(row, portfolio)

        if base_signal == 0:
            return 0, "none"
            
        confirmation_count = 0
        if base_signal == 1:
            for period in self.rolling_periods:
                if getattr(row, f'tsi_{period}', 0) >= self.strength_threshold:
                    confirmation_count += 1
        elif base_signal == -1:
            for period in self.rolling_periods:
                if getattr(row, f'tsi_{period}', 0) <= -self.strength_threshold:
                    confirmation_count += 1
        
        if confirmation_count >= self.min_confirmation:
            signal_type = f"{base_signal_type}_roll_bull" if base_signal == 1 else f"{base_signal_type}_roll_bear"
            return base_signal, signal_type
        else:
            return 0, "rolling_filtered"


class RollingCandleATRStrategy(BandVWAPStrategy):
    def __init__(self, data, params):
        self.volatility_thresholds = params.get('volatility_thresholds', {'low': 0.7, 'high': 1.5})
        self.regime_lookbacks = params.get('regime_lookbacks', {
            'low': [120, 180, 240],
            'normal': [60, 120, 180],
            'high': [15, 30, 60]
        })
        self.strength_threshold = params.get('strength_threshold', 0.7)
        self.min_confirmation = params.get('min_confirmation', 2)
        super().__init__(data, params)
    
    def _calculate_bands(self):
        super()._calculate_bands()
        self._calculate_rolling_candles()

    def _calculate_rolling_candles(self):
        df = self.data
        all_periods = set()
        for periods in self.regime_lookbacks.values():
            all_periods.update(periods)

        for period in sorted(list(all_periods)):
            df[f'roll_{period}_open'] = df['open'].shift(period - 1)
            df[f'roll_{period}_high'] = df['high'].rolling(window=period).max()
            df[f'roll_{period}_low'] = df['low'].rolling(window=period).min()
            df[f'roll_{period}_close'] = df['close']

            roll_range = df[f'roll_{period}_high'] - df[f'roll_{period}_low']
            roll_body = df[f'roll_{period}_close'] - df[f'roll_{period}_open']
            
            df[f'tsi_{period}'] = (roll_body / roll_range).replace([np.inf, -np.inf], 0).fillna(0)

    def get_entry_signal(self, row, portfolio):
        base_signal, base_signal_type = super().get_entry_signal(row, portfolio)
        if base_signal == 0:
            return 0, "none"
        
        if pd.isna(row.atr) or pd.isna(row.historical_avg_atr) or row.historical_avg_atr == 0:
            return 0, "atr_data_missing"
            
        ratio = row.atr / row.historical_avg_atr
        
        if ratio < self.volatility_thresholds['low']:
            regime = 'low'
        elif ratio > self.volatility_thresholds['high']:
            regime = 'high'
        else:
            regime = 'normal'
            
        lookback_periods = self.regime_lookbacks[regime]
        
        confirmation_count = 0
        if base_signal == 1:
            for period in lookback_periods:
                if getattr(row, f'tsi_{period}', 0) >= self.strength_threshold:
                    confirmation_count += 1
        elif base_signal == -1:
            for period in lookback_periods:
                if getattr(row, f'tsi_{period}', 0) <= -self.strength_threshold:
                    confirmation_count += 1

        if confirmation_count >= self.min_confirmation:
            return base_signal, f"{base_signal_type}_atr_bull_{regime}" 
        return 0, f"dynamic_atr_filtered_{regime}"

class Backtester:
    def __init__(self, data: pd.DataFrame, portfolio: Portfolio, strategy: Strategy, params: dict):
        self.data = data
        self.portfolio = portfolio
        self.strategy = strategy
        self.params = params

    def run(self):
        for day, day_data in tqdm(self.data.groupby(self.data.index.date), desc="Backtesting"):
            self.portfolio.record_daily_state(day)
            daily_trade_amount = self.portfolio.history.loc[day, 'aum']
            
            for i, row in enumerate(day_data.itertuples()):
                is_eod = i == len(day_data) - 1
                is_trade_time = (row.min_from_open >= 0 and row.min_from_open % self.params['trade_freq'] == 0)

                if self.portfolio.position['direction'] != 0:
                    action = self.strategy.get_exit_action(row, self.portfolio)
                    if action or is_eod:
                        if action and 'reason' in action:
                            reason = action['reason']
                            if reason == 'ladder_stop':
                                exit_reason = f"L{action['level']} Stop"
                            elif reason == 'ladder_profit':
                                level = action['level']
                                size_pct = int(action.get('size', 1.0) * 100)
                                exit_reason = f"L{level} Profit ({size_pct}%)"
                            elif reason == 'band_reentry':
                                exit_reason = "Band Reentry"
                            elif reason == 'vwap_exit':
                                exit_reason = "VWAP Exit"
                            elif reason == 'atr_stop':
                                exit_reason = "ATR Stop"
                            elif reason == 'atr_profit':
                                exit_reason = "ATR Profit"
                            else:
                                exit_reason = reason
                        elif is_eod:
                            exit_reason = "EOD"
                        else:
                            exit_reason = "Unknown"
                        
                        self.portfolio.execute_exit(row.close, row.Index, action or {'type': 'full'}, exit_reason)
                        if is_eod: continue
                
                if is_trade_time:
                    signal, signal_type = self.strategy.get_entry_signal(row, self.portfolio)
                    if signal != 0:
                        self.portfolio.execute_entry(signal, row.close, row.Index, daily_trade_amount, signal_type)
        
        history = self.portfolio.history.copy()
        history['daily_return'] = history['aum'].pct_change().fillna(0)
        history['daily_gross_return'] = history['gross_aum'].pct_change().fillna(0)
        history['cumulative_return'] = (history['aum'] / self.portfolio.initial_aum) - 1
        history['cumulative_gross_return'] = (history['gross_aum'] / self.portfolio.initial_aum) - 1
        return history, pd.DataFrame(self.portfolio.trades)

class Reporter:
    def __init__(self, daily_results, trades, initial_aum, feature_data=None):
        self.daily_results = daily_results
        self.trades = trades
        self.initial_aum = initial_aum
        self.feature_data = feature_data
        self.strategy_params = {}

    def _calculate_drawdowns(self):
        if self.daily_results.empty:
            return None, None, None, None
            
        cumulative_returns = self.daily_results['aum'] / self.initial_aum
        running_max = cumulative_returns.cummax()
        
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        gross_cumulative_returns = self.daily_results['gross_aum'] / self.initial_aum
        gross_running_max = gross_cumulative_returns.cummax()
        gross_drawdowns = (gross_cumulative_returns - gross_running_max) / gross_running_max
        gross_max_drawdown = gross_drawdowns.min()
        
        return drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown

    def print_summary(self):
        if self.daily_results.empty: print("No results to summarize."); return
        
        # Calculate total returns
        total_net_return = (self.daily_results['aum'].iloc[-1] / self.initial_aum) - 1
        total_gross_return = (self.daily_results['gross_aum'].iloc[-1] / self.initial_aum) - 1
        total_trades = len(self.trades)
        
        # Calculate CAGR
        start_date = self.daily_results.index[0]
        end_date = self.daily_results.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25 if days > 0 else 1
        
        net_cagr = ((self.daily_results['aum'].iloc[-1] / self.initial_aum) ** (1/years) - 1) if years > 0 else 0
        gross_cagr = ((self.daily_results['gross_aum'].iloc[-1] / self.initial_aum) ** (1/years) - 1) if years > 0 else 0
        
        # Calculate maximum drawdowns
        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        
        summary = {
            "Total Net Return": f"{total_net_return:.2%}",
            "Total Gross Return": f"{total_gross_return:.2%}",
            "CAGR (Net)": f"{net_cagr:.2%}",
            "CAGR (Gross)": f"{gross_cagr:.2%}",
            "Maximum Drawdown (Net)": f"{max_drawdown:.2%}" if max_drawdown is not None else "N/A",
            "Maximum Drawdown (Gross)": f"{gross_max_drawdown:.2%}" if gross_max_drawdown is not None else "N/A",
            "Total Trades": total_trades,
        }
        
        if not self.trades.empty:
            # Calculate win rate and average returns
            win_rate = (self.trades['net_pnl'] > 0).mean()
            avg_gross_return = self.trades['gross_return'].mean()
            avg_net_return = self.trades['net_return'].mean()
            
            # Calculate commission metrics
            total_commission = self.trades['commission'].sum()
            avg_commission_per_trade = total_commission / total_trades if total_trades > 0 else 0
            
            # Calculate profit/loss ratio
            winning_trades = self.trades[self.trades['net_return'] > 0]['net_return']
            losing_trades = self.trades[self.trades['net_return'] < 0]['net_return']
            
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                avg_win = winning_trades.mean()
                avg_loss = abs(losing_trades.mean())
                profit_loss_ratio = avg_win / avg_loss
            else:
                profit_loss_ratio = None
        
            signal_breakdown = self.trades['signal_type'].value_counts()
            
            summary.update({
                "Win Rate": f"{win_rate:.2%}",
                "Avg Gross Return/Trade": f"{avg_gross_return:.2%}",
                "Avg Net Return/Trade": f"{avg_net_return:.2%}",
                "Profit/Loss Ratio": f"{profit_loss_ratio:.2f}" if profit_loss_ratio is not None else "N/A",
                "Total Commission": f"{total_commission:,.0f}",
                "Avg Commission/Trade": f"{avg_commission_per_trade:,.0f}",
            })
        else:
            summary["Win Rate"] = "N/A"
        
        print("\n--- Backtest Summary ---")
        [print(f"{k:<25}: {v}") for k, v in summary.items()]
        
        if not self.trades.empty and len(signal_breakdown) > 0:
            print(f"\n--- Signal Type Breakdown ---")
            for signal_type, count in signal_breakdown.items():
                pct = count / total_trades * 100
                avg_return = self.trades[self.trades['signal_type'] == signal_type]['net_return'].mean()
                print(f"{signal_type:<20}: {count:>3} trades ({pct:>5.1f}%) | Avg Return: {avg_return:>6.2%}")
                
        print("-" * 50)

    def plot_aum_curve(self):
        if self.daily_results.empty: return
        
        self.daily_results.index = pd.to_datetime(self.daily_results.index)

        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 18), constrained_layout=True)
        
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.5], hspace=0.2)
        gs_top = gs[0].subgridspec(3, 1, hspace=0)
        
        ax1 = fig.add_subplot(gs_top[0])
        ax2 = fig.add_subplot(gs_top[1], sharex=ax1)
        ax3 = fig.add_subplot(gs_top[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[1])

        ax1.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='x', labelbottom=False)

        self.daily_results['gross_aum'].plot(ax=ax1, label='AUM (Gross, Pre-Cost)', color='green', linestyle='--')
        self.daily_results['aum'].plot(ax=ax1, label='AUM (Net, Post-Cost)', color='blue')
        
        if not self.daily_results.empty and not self.daily_results['aum'].empty:
            start_date, end_date = self.daily_results.index[0], self.daily_results.index[-1]
            total_days = (end_date - start_date).days
            if total_days <= 0:
                total_days = 1

            points_to_annotate = {}
            
            peak_aum_date = self.daily_results['aum'].idxmax()
            points_to_annotate['peak_aum'] = {
                'date': peak_aum_date,
                'value': self.daily_results.loc[peak_aum_date, 'aum'],
                'text': f'Peak AUM\n{self.daily_results.loc[peak_aum_date, "aum"]:,.0f}\n{peak_aum_date.strftime("%Y-%m-%d")}'
            }
            trough_aum_date = self.daily_results['aum'].idxmin()
            points_to_annotate['trough_aum'] = {
                'date': trough_aum_date,
                'value': self.daily_results.loc[trough_aum_date, 'aum'],
                'text': f'Trough AUM\n{self.daily_results.loc[trough_aum_date, "aum"]:,.0f}\n{trough_aum_date.strftime("%Y-%m-%d")}'
            }

            if 'daily_return' in self.daily_results.columns:
                best_return_date = self.daily_results['daily_return'].idxmax()
                best_return_val = self.daily_results.loc[best_return_date, 'daily_return']
                points_to_annotate['best_return'] = {
                    'date': best_return_date,
                    'value': self.daily_results.loc[best_return_date, 'aum'],
                    'text': f'Best Return\n{best_return_val:+.2%}\n{best_return_date.strftime("%Y-%m-%d")}'
                }
                worst_return_date = self.daily_results['daily_return'].idxmin()
                worst_return_val = self.daily_results.loc[worst_return_date, 'daily_return']
                points_to_annotate['worst_return'] = {
                    'date': worst_return_date,
                    'value': self.daily_results.loc[worst_return_date, 'aum'],
                    'text': f'Worst Return\n{worst_return_val:+.2%}\n{worst_return_date.strftime("%Y-%m-%d")}'
                }
            
            annotation_config = {
                'peak_aum':     {'va': 'bottom', 'offset_y': 15,   'color': 'green',  'marker': 'o'},
                'trough_aum':   {'va': 'top',    'offset_y': -15,  'color': 'red',    'marker': 'o'},
                'best_return':  {'va': 'bottom', 'offset_y': 40,   'color': 'blue',   'marker': '*'},
                'worst_return': {'va': 'top',    'offset_y': -40,  'color': 'purple', 'marker': '*'},
            }

            for key, point in points_to_annotate.items():
                config = annotation_config.get(key)
                if not config: continue
                
                is_left_half = (point['date'] - start_date).days < total_days / 2
                ha = 'left' if is_left_half else 'right'
                xytext_x = 50 if is_left_half else -50
                connection_rad = 0.2 if is_left_half else -0.2
                
                plot_date = point['date'].to_pydatetime()

                ax1.scatter(plot_date, point['value'], marker=config['marker'], s=150, facecolors='none', edgecolors=config['color'], linewidths=2.5, zorder=5)
                ax1.annotate(point['text'],
                             xy=(plot_date, point['value']),
                             xytext=(xytext_x, config['offset_y']),
                             textcoords='offset points',
                             ha=ha,
                             va=config['va'],
                             arrowprops=dict(arrowstyle='-|>,' + f'head_width=0.4,' + f'head_length=0.8',
                                             connectionstyle=f'arc3,rad={connection_rad}',
                                             color=config['color'],
                                             lw=1.5),
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=config['color'], lw=1, alpha=0.8))

        price_drawdowns = None
        max_price_dd = None
        
        if self.feature_data is not None and not self.feature_data.empty:
            daily_close = self.feature_data.groupby(self.feature_data.index.date)['close'].last()
            
            daily_results_datetime_index = self.daily_results.index
            daily_close = daily_close[daily_close.index.isin(daily_results_datetime_index.date)]
            
            if not daily_close.empty:
                ax1_twin = ax1.twinx()
                
                initial_price = daily_close.iloc[0]
                close_cumulative_return = (daily_close / initial_price - 1) * 100
                close_cumulative_return.plot(ax=ax1_twin, label='Buy&Hold Return (%)', color='gray', alpha=0.8, linewidth=1.5)
                ax1_twin.set_ylabel('Buy&Hold Cumulative Return (%)', color='gray')
                ax1_twin.tick_params(axis='y', labelcolor='gray')
                
                price_returns = daily_close / initial_price
                price_running_max = price_returns.cummax()
                price_drawdowns = (price_returns - price_running_max)
                max_price_dd = price_drawdowns.min()
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax1_twin.legend().set_visible(False)
        
        ax1.set_title('AUM and Buy&Hold Cumulative Return Over Time', fontsize=16)
        ax1.set_ylabel('AUM')
        if self.feature_data is None or self.feature_data.empty:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if drawdowns is not None:
            gross_drawdowns.plot(ax=ax2, label=f'Gross DD (Max: {gross_max_drawdown:.2%})', color='red', linestyle='--', alpha=0.7)
            drawdowns.plot(ax=ax2, label=f'Net DD (Max: {max_drawdown:.2%})', color='darkred')
            ax2.fill_between(drawdowns.index, drawdowns, 0, alpha=0.3, color='red')
            
            if price_drawdowns is not None:
                ax2_twin = ax2.twinx()
                price_drawdowns.plot(ax=ax2_twin, label=f'Buy&Hold DD (Max: {max_price_dd:.2%})', color='gray', alpha=0.8, linewidth=1.5)
                ax2_twin.set_ylabel('Buy&Hold Drawdown (%)', color='gray')
                ax2_twin.tick_params(axis='y', labelcolor='gray')
                ax2_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
                ax2_twin.legend().set_visible(False)
            
            ax2.set_title('Strategy vs Buy&Hold Drawdown Over Time', fontsize=14)
            ax2.set_ylabel('Strategy Drawdown (%)')
            if price_drawdowns is None:
                ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        if self.feature_data is not None and not self.feature_data.empty and 'open' in self.feature_data.columns:
            daily_agg = self.feature_data.groupby(self.feature_data.index.date).agg(
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last')
            )
            daily_agg.index = pd.to_datetime(daily_agg.index)
            
            intraday_power = ((daily_agg['close'] - daily_agg['open']).abs() / 
                              (daily_agg['high'] - daily_agg['low']).abs())
            intraday_power.replace([np.inf, -np.inf], np.nan, inplace=True)
            intraday_power.fillna(0, inplace=True)
            
            daily_results_datetime_index = self.daily_results.index
            intraday_power = intraday_power[intraday_power.index.isin(daily_results_datetime_index)]
            
            if not intraday_power.empty:
                ax3.bar(intraday_power.index, intraday_power, color='purple', alpha=0.6, width=1.0)
                
                ax3.set_title('Intraday Power Ratio (|Close-Open| / |High-Low|)', fontsize=14)
                ax3.set_ylabel('Ratio')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(bottom=0)
                ax3.set_xlabel('Date')
                ax3.tick_params(axis='x', rotation=45)

                daily_returns = self.daily_results['daily_return'].copy()
                combined_data = pd.DataFrame({'power_ratio': intraday_power, 'daily_return': daily_returns}).dropna()
                combined_data.index = pd.to_datetime(combined_data.index)

                if len(combined_data) >= 5:
                    try:
                        labels = ['Strong Loss', 'Loss', 'Neutral', 'Profit', 'Strong Profit']
                        
                        ranks = combined_data['daily_return'].rank(method='first')
                        combined_data['return_category'] = pd.qcut(ranks, q=5, labels=labels)
                        
                        gb = combined_data.groupby('return_category', observed=True)['daily_return']
                        avg_power_by_return = combined_data.groupby('return_category', observed=True)['power_ratio'].mean()
                        
                        if isinstance(avg_power_by_return.index, pd.CategoricalDtype):
                            avg_power_by_return = avg_power_by_return.reindex(labels)

                        colors = plt.get_cmap('gray_r')(np.linspace(0.1, 0.7, len(labels)))
                        avg_power_by_return.plot(kind='bar', ax=ax4, color=colors, alpha=0.8)
                        
                        bins_min = gb.min()
                        bins_max = gb.max()
                        bin_labels = [f"{bins_min[label]:.2%} to {bins_max[label]:.2%}" for label in avg_power_by_return.index if label in bins_min]

                        for i, patch in enumerate(ax4.patches):
                            if i < len(bin_labels):
                                ax4.text(patch.get_x() + patch.get_width() / 2.,
                                         patch.get_height(),
                                         bin_labels[i],
                                         ha='center',
                                         va='bottom',
                                         fontsize=9,
                                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.6, edgecolor='none'))
                        
                        ax4.set_title('Average Power Ratio by Daily Return Quintile', fontsize=14)
                        ax4.set_ylabel('Average Ratio')
                        ax4.set_xlabel('Return Category')
                        ax4.tick_params(axis='x', rotation=0)
                        ax4.grid(True, alpha=0.3, axis='y')

                    except Exception as e:
                        print(f"An unexpected error occurred while creating return quantiles: {e}")
                else:
                    print("Not enough daily returns to create 5 quantiles for analysis.")

        plt.show()

    def plot_candlestick_performance(self):
        if self.trades.empty or self.feature_data is None:
            print("Not enough data for candlestick performance analysis.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        daily_ohlc = self.feature_data.groupby(self.feature_data.index.date).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last')
        )
        daily_ohlc.index = pd.to_datetime(daily_ohlc.index)

        analyzer = CandlestickAnalyzer()
        daily_ohlc['pattern'] = analyzer.classify(daily_ohlc)
        
        trades_df = self.trades.copy()
        trades_df['entry_day'] = pd.to_datetime(trades_df['entry_time']).dt.normalize()
        
        merged_df = pd.merge(trades_df, daily_ohlc[['pattern']], left_on='entry_day', right_index=True)
        
        if merged_df.empty:
            print("No trades could be matched with candlestick patterns.")
            return
            
        performance = merged_df.groupby('pattern')['net_return'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        if performance.empty:
            print("No performance data to plot for candlestick patterns.")
            return

        korean_patterns = {
            "Doji": "십자형",
            "Bullish Marubozu": "장대양봉",
            "Bearish Marubozu": "장대음봉",
            "Hammer / Hanging Man": "망치형/교수형",
            "Shooting Star": "유성형",
            "Spinning Top": "팽이형",
            "Standard Candle": "보통형"
        }
        performance.rename(index=korean_patterns, inplace=True)

        fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
        
        colors = ['green' if x > 0 else 'red' for x in performance['mean']]
        
        bars = sns.barplot(x=performance.index, y=performance['mean'], ax=ax, palette=colors)
        
        for i, bar in enumerate(bars.patches):
            mean_return = performance['mean'].iloc[i]
            trade_count = performance['count'].iloc[i]
            label = f"{mean_return:.2%}\n({trade_count} trades)"
            
            y_pos = bar.get_height()
            va = 'bottom' if mean_return >= 0 else 'top'
            offset = 0.0005 if mean_return >=0 else -0.0005
            
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset, label,
                    ha='center', va=va,
                    fontsize=10, color='black')
                    
        ax.set_title('Average Net Return by Candlestick Pattern on Entry Day', fontsize=16)
        ax.set_ylabel('Average Net Return (%)')
        ax.set_xlabel('캔들 패턴')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.show()

    def plot_drawdown_analysis(self):
        if self.daily_results.empty:
            print("No data available for drawdown analysis.")
            return
            
        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        if drawdowns is None:
            print("Unable to calculate drawdowns.")
            return
            
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        drawdowns.plot(ax=ax1, color='red', alpha=0.8)
        ax1.fill_between(drawdowns.index, drawdowns, 0, alpha=0.3, color='red')
        ax1.set_title(f'Net Drawdown (Max: {max_drawdown:.2%})')
        ax1.set_ylabel('Drawdown')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax1.grid(True, alpha=0.3)
        
        drawdowns.hist(bins=50, ax=ax2, alpha=0.7, color='red')
        ax2.axvline(max_drawdown, color='darkred', linestyle='--', linewidth=2, label=f'Max DD: {max_drawdown:.2%}')
        ax2.set_title('Drawdown Distribution')
        ax2.set_xlabel('Drawdown')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        cumulative_returns = self.daily_results['aum'] / self.initial_aum
        running_max = cumulative_returns.cummax()
        
        (cumulative_returns - 1).plot(ax=ax3, label='Cumulative Return', color='blue')
        (running_max - 1).plot(ax=ax3, label='Running Peak', color='green', linestyle='--')
        ax3.fill_between(cumulative_returns.index, cumulative_returns - 1, running_max - 1, alpha=0.3, color='red')
        ax3.set_title('Underwater Plot')
        ax3.set_ylabel('Return')
        ax3.legend()
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax3.grid(True, alpha=0.3)
        
        rolling_dd = drawdowns.rolling(window=30).min()
        rolling_dd.plot(ax=ax4, color='orange', label='30-Day Rolling Max DD')
        ax4.set_title('Rolling Maximum Drawdown (30 Days)')
        ax4.set_ylabel('Max Drawdown')
        ax4.set_xlabel('Date')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.suptitle(f'Drawdown Analysis - MDD: {max_drawdown:.2%}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

    def plot_intraday(self, target_date: str):
        if self.feature_data is None:
            print("Feature data not available. Please pass feature_data to Reporter.")
            return
            
        try:
            target_date_obj = pd.to_datetime(target_date).date()
        except:
            print(f"Invalid date format: {target_date}. Please use 'YYYY-MM-DD' format.")
            return

        available_days = sorted(self.feature_data['day'].unique())
        
        if target_date_obj not in available_days:
            print(f"Date {target_date} is not a trading day or no data available.")
            before_dates = [d for d in available_days if d < target_date_obj][-5:]
            after_dates = [d for d in available_days if d > target_date_obj][:5]
            print("Nearby trading days:")
            if before_dates: print(f"  Before: {', '.join([str(d) for d in before_dates])}")
            if after_dates: print(f"  After: {', '.join([str(d) for d in after_dates])}")
            return

        day_data = self.feature_data[self.feature_data['day'] == target_date_obj].copy()
        if day_data.empty:
            print(f"No data available for date {target_date}.")
            return

        day_trades = []
        if not self.trades.empty:
            for _, trade in self.trades.iterrows():
                entry_date = pd.to_datetime(trade['entry_time']).date()
                exit_date = pd.to_datetime(trade['exit_time']).date()
                if entry_date == target_date_obj or exit_date == target_date_obj:
                    day_trades.append(trade)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        time_labels = day_data.index.strftime('%H:%M')
        x_pos = range(len(day_data))

        ax.plot(x_pos, day_data['close'], label='Close Price', color='black', linewidth=2.5, zorder=3)
        
        if 'vwap' in day_data.columns and day_data['vwap'].notna().any():
            ax.plot(x_pos, day_data['vwap'], label='VWAP', color='purple', linewidth=2, alpha=0.8, zorder=2)
        
        if 'UB' in day_data.columns and day_data['UB'].notna().any():
            ax.plot(x_pos, day_data['UB'], label='Entry Upper Band', color='red', linestyle='-', alpha=0.8, linewidth=1.5)
            ax.plot(x_pos, day_data['LB'], label='Entry Lower Band', color='blue', linestyle='-', alpha=0.8, linewidth=1.5)
            ax.fill_between(x_pos, day_data['UB'], day_data['LB'], alpha=0.08, color='gray', label='Entry Band Zone')

        if 'exit_UB' in day_data.columns and day_data['exit_UB'].notna().any():
            ax.plot(x_pos, day_data['exit_UB'], label='Exit Upper Band', color='darkred', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.plot(x_pos, day_data['exit_LB'], label='Exit Lower Band', color='darkblue', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.fill_between(x_pos, day_data['exit_UB'], day_data['exit_LB'], alpha=0.05, color='orange', label='Exit Band Zone')

        trade_events = []
        trade_colors = ['green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
        
        day_trades_sorted = sorted(day_trades, key=lambda x: pd.to_datetime(x['entry_time']))
        
        for trade_idx, trade in enumerate(day_trades_sorted):
            color_idx = trade_idx % len(trade_colors)
            base_color = trade_colors[color_idx]
            
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            entry_idx = None
            exit_idx = None
            
            if entry_time.date() == target_date_obj:
                entry_idx = day_data.index.get_indexer([entry_time], method='nearest')[0]
                if entry_idx >= 0:
                    direction = 'long' if trade['direction'] > 0 else 'short'
                    marker = '^' if direction == 'long' else 'v'
                    ax.scatter(entry_idx, trade['entry_price'], color=base_color, marker=marker, s=150, zorder=6, edgecolors='white', linewidth=1)
                    signal_type = trade.get('signal_type', 'unknown')
                    trade_events.append(f"T{trade_idx+1}: {entry_time.strftime('%H:%M')} - {direction.upper()} Entry ({signal_type}): {trade['entry_price']:.0f}")
            
            if exit_time.date() == target_date_obj:
                exit_idx = day_data.index.get_indexer([exit_time], method='nearest')[0]
                if exit_idx >= 0:
                    ax.scatter(exit_idx, trade['exit_price'], color=base_color, marker='x', s=150, zorder=6, linewidth=3)
                    net_return = trade.get('net_return', 0)
                    commission = trade.get('commission', 0)
                    exit_reason = trade.get('exit_reason', 'Unknown')
                    trade_events.append(f"T{trade_idx+1}: {exit_time.strftime('%H:%M')} - Exit ({exit_reason}): {trade['exit_price']:.0f} | Return: {net_return:.2%} | Cost: {commission:.0f}")

            if entry_idx is not None:
                start_x = entry_idx
                end_x = exit_idx if exit_idx is not None else len(x_pos) - 1

                # ATR Stop/Profit Lines
                if 'atr_multiplier_stop' in self.strategy_params:
                    entry_atr = day_data.iloc[entry_idx]['atr']
                    atr_stop = self.strategy_params.get('atr_multiplier_stop', 2.0)
                    atr_profit = self.strategy_params.get('atr_multiplier_profit', 4.0)
                    
                    stop_price = trade['entry_price'] - trade['direction'] * entry_atr * atr_stop
                    profit_price = trade['entry_price'] + trade['direction'] * entry_atr * atr_profit
                    
                    ax.hlines(stop_price, start_x, end_x, colors='darkred', linestyles='-.', alpha=0.9, linewidth=1.5,
                              label=f'ATR Stop ({atr_stop}x)' if trade_idx == 0 else "")
                    ax.text(start_x + 2, stop_price, f'T{trade_idx+1} ATR Stop', fontsize=8, color='darkred',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    ax.hlines(profit_price, start_x, end_x, colors='darkgreen', linestyles='-.', alpha=0.9, linewidth=1.5,
                              label=f'ATR Profit ({atr_profit}x)' if trade_idx == 0 else "")
                    ax.text(start_x + 2, profit_price, f'T{trade_idx+1} ATR Profit', fontsize=8, color='darkgreen',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

                # Ladder Stop/Profit Lines
                if 'ladder_levels' in self.strategy_params:
                    ladder_levels = self.strategy_params.get('ladder_levels', [])
                    entry_price = trade['entry_price']
                    direction = trade['direction']
                    
                    for i, level in enumerate(ladder_levels):
                        stop_price = entry_price * (1 + level['stop'] * direction)
                        profit_price = entry_price * (1 + level['profit'] * direction)
                        
                        ax.hlines(stop_price, start_x, end_x, colors='red', linestyles=':', alpha=0.8, linewidth=2, 
                                 label=f'L{i} Stop ({level["stop"]*100:+.1f}%)' if trade_idx == 0 and i == 0 else "")
                        ax.text(start_x + 2, stop_price, f'T{trade_idx+1}L{i}S', fontsize=8, color='red', 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                        
                        ax.hlines(profit_price, start_x, end_x, colors='green', linestyles=':', alpha=0.8, linewidth=2,
                                 label=f'L{i} Profit ({level["profit"]*100:+.1f}%)' if trade_idx == 0 and i == 0 else "")
                        ax.text(start_x + 2, profit_price, f'T{trade_idx+1}L{i}P({level["size"]*100:.0f}%)', fontsize=8, color='green',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        daily_pnl = None
        daily_return = None
        if target_date_obj in self.daily_results.index:
            daily_pnl = self.daily_results.loc[target_date_obj, 'pnl']
            if 'daily_return' in self.daily_results.columns:
                daily_return = self.daily_results.loc[target_date_obj, 'daily_return']

        title = f'Intraday Chart - {target_date}'
        if daily_pnl is not None:
            title += f' | Daily PnL: {daily_pnl:,.0f}'
            if daily_return is not None:
                title += f' ({daily_return:.2%})'
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        
        tick_interval = max(1, len(day_data) // 15)
        ax.set_xticks(x_pos[::tick_interval])
        ax.set_xticklabels(time_labels[::tick_interval], rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        if trade_events:
            print(f"\n=== {target_date} Trade Summary ===")
            if daily_pnl is not None:
                print(f"Daily PnL: {daily_pnl:,.0f}")
                if daily_return is not None:
                    print(f"Daily Return: {daily_return:.2%}")
            
            print("\nTrade Details:")
            for event in trade_events:
                print(f"  {event}")
                
            entry_trades = [event for event in trade_events if 'Entry' in event]
            exit_trades = [event for event in trade_events if 'Exit' in event]
            
            print(f"\n=== Daily Trading Statistics ===")
            print(f"  Entries: {len(entry_trades)} | Exits: {len(exit_trades)}")
            
            exit_reasons = {}
            for event in exit_trades:
                if '(' in event and ')' in event:
                    reason_start = event.find('Exit (') + 6
                    reason_end = event.find(')', reason_start)
                    if reason_start > 5 and reason_end > reason_start:
                        reason = event[reason_start:reason_end]
                        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            if exit_reasons:
                print(f"  Exit Reasons:")
                for reason, count in exit_reasons.items():
                    print(f"    {reason}: {count}")
            
            day_trade_returns = [trade.get('net_return', 0) for trade in day_trades]
            day_trade_commissions = [trade.get('commission', 0) for trade in day_trades]
            
            if day_trade_returns:
                print(f"\n=== Performance Metrics ===")
                print(f"  Total Completed Trades: {len(day_trades)}")
                print(f"  Winning Trades: {sum(1 for r in day_trade_returns if r > 0)}")
                print(f"  Losing Trades: {sum(1 for r in day_trade_returns if r < 0)}")
                print(f"  Win Rate: {sum(1 for r in day_trade_returns if r > 0) / len(day_trade_returns):.1%}")
                print(f"  Average Return: {np.mean(day_trade_returns):.2%}")
                print(f"  Best Trade: {max(day_trade_returns):.2%}")
                print(f"  Worst Trade: {min(day_trade_returns):.2%}")
                print(f"  Total Commission: {sum(day_trade_commissions):,.0f}")
                
            if hasattr(self, 'ladder_levels') and self.ladder_levels:
                print(f"\n=== Ladder Level Analysis ===")
                for trade_idx, trade in enumerate(day_trades_sorted):
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    
                    if entry_time.date() == target_date_obj:
                        entry_price = trade['entry_price']
                        direction = trade['direction']
                        exit_reason = trade.get('exit_reason', 'Unknown')
                        
                        trade_data = day_data[(day_data.index >= entry_time) & (day_data.index <= exit_time)]
                        if not trade_data.empty:
                            min_price = trade_data['low'].min()
                            max_price = trade_data['high'].max()
                            
                            print(f"\nT{trade_idx+1} ({entry_time.strftime('%H:%M')}-{exit_time.strftime('%H:%M')}) Exit: {exit_reason}")
                            print(f"  Entry: {entry_price:.0f}, Exit: {trade['exit_price']:.0f}")
                            print(f"  Price Range: {min_price:.0f} - {max_price:.0f}")
                            
                            for i, level in enumerate(self.ladder_levels):
                                stop_price = entry_price * (1 + level['stop'] * direction)
                                profit_price = entry_price * (1 + level['profit'] * direction)
                                
                                if direction == 1:  # Long
                                    stop_reached = min_price <= stop_price
                                    profit_reached = max_price >= profit_price
                                else:  # Short
                                    stop_reached = max_price >= stop_price
                                    profit_reached = min_price <= profit_price
                                
                                print(f"  L{i} Stop {stop_price:.0f}: {'✓' if stop_reached else '✗'}")
                                print(f"  L{i} Profit {profit_price:.0f}: {'✓' if profit_reached else '✗'}")
        else:
            print(f"\nNo trades occurred on {target_date}.")
            if daily_pnl is not None:
                print(f"Daily PnL: {daily_pnl:,.0f}")
                if daily_return is not None:
                    print(f"Daily Return: {daily_return:.2%}")


class CandlestickAnalyzer:
    def __init__(self,
                 body_thresh: float = 0.8,
                 wick_ratio_thresh: float = 2.0,
                 doji_thresh: float = 0.1,
                 spinning_top_body_thresh: float = 0.3):
        self.params = {
            'body_thresh': body_thresh,
            'wick_ratio_thresh': wick_ratio_thresh,
            'doji_thresh': doji_thresh,
            'spinning_top_body_thresh': spinning_top_body_thresh,
        }

    def classify(self, ohlc_df: pd.DataFrame) -> pd.Series:
        return ohlc_df.apply(self._get_pattern, axis=1)

    def _get_pattern(self, row: pd.Series) -> str:
        body = abs(row.close - row.open)
        total_range = row.high - row.low

        if total_range < 1e-9:
            return "Doji"

        body_ratio = body / total_range
        upper_wick = row.high - max(row.open, row.close)
        lower_wick = min(row.open, row.close) - row.low

        is_bullish = row.close > row.open

        if body_ratio < self.params['doji_thresh']:
            return "Doji"

        if body_ratio > self.params['body_thresh']:
            return "Bullish Marubozu" if is_bullish else "Bearish Marubozu"
        
        body_not_zero = body > 1e-9
        
        if body_not_zero and lower_wick / body > self.params['wick_ratio_thresh'] and upper_wick / total_range < self.params['doji_thresh']:
            return "Hammer / Hanging Man"

        if body_not_zero and upper_wick / body > self.params['wick_ratio_thresh'] and lower_wick / total_range < self.params['doji_thresh']:
            return "Shooting Star"

        if body_ratio < self.params['spinning_top_body_thresh']:
            return "Spinning Top"
        return "Standard Candle"

class RollingCandleStrategy(BandVWAPStrategy):
    def __init__(self, data, params):
        self.rolling_periods = params.get('rolling_periods', [30, 60, 120])
        self.strength_threshold = params.get('strength_threshold', 0.7)
        self.min_confirmation = params.get('min_confirmation', 2)
        super().__init__(data, params)

    def _calculate_bands(self):
        super()._calculate_bands()
        self._calculate_rolling_candles()

    def _calculate_rolling_candles(self):
        df = self.data
        for period in self.rolling_periods:
            df[f'roll_{period}_open'] = df['open'].shift(period - 1)
            df[f'roll_{period}_high'] = df['high'].rolling(window=period).max()
            df[f'roll_{period}_low'] = df['low'].rolling(window=period).min()
            df[f'roll_{period}_close'] = df['close']

            roll_range = df[f'roll_{period}_high'] - df[f'roll_{period}_low']
            roll_body = df[f'roll_{period}_close'] - df[f'roll_{period}_open']
            
            df[f'tsi_{period}'] = (roll_body / roll_range).replace([np.inf, -np.inf], 0).fillna(0)

    def get_entry_signal(self, row, portfolio):
        base_signal, base_signal_type = super().get_entry_signal(row, portfolio)

        if base_signal == 0:
            return 0, "none"
            
        confirmation_count = 0
        if base_signal == 1:
            for period in self.rolling_periods:
                if getattr(row, f'tsi_{period}', 0) >= self.strength_threshold:
                    confirmation_count += 1
        elif base_signal == -1:
            for period in self.rolling_periods:
                if getattr(row, f'tsi_{period}', 0) <= -self.strength_threshold:
                    confirmation_count += 1
        
        if confirmation_count >= self.min_confirmation:
            signal_type = f"{base_signal_type}_roll_bull" if base_signal == 1 else f"{base_signal_type}_roll_bear"
            return base_signal, signal_type
        else:
            return 0, "rolling_filtered"


class RollingCandleATRStrategy(BandVWAPStrategy):
    def __init__(self, data, params):
        self.volatility_thresholds = params.get('volatility_thresholds', {'low': 0.7, 'high': 1.5})
        self.regime_lookbacks = params.get('regime_lookbacks', {
            'low': [120, 180, 240],
            'normal': [60, 120, 180],
            'high': [15, 30, 60]
        })
        self.strength_threshold = params.get('strength_threshold', 0.7)
        self.min_confirmation = params.get('min_confirmation', 2)
        super().__init__(data, params)
    
    def _calculate_bands(self):
        super()._calculate_bands()
        self._calculate_rolling_candles()

    def _calculate_rolling_candles(self):
        df = self.data
        all_periods = set()
        for periods in self.regime_lookbacks.values():
            all_periods.update(periods)

        for period in sorted(list(all_periods)):
            df[f'roll_{period}_open'] = df['open'].shift(period - 1)
            df[f'roll_{period}_high'] = df['high'].rolling(window=period).max()
            df[f'roll_{period}_low'] = df['low'].rolling(window=period).min()
            df[f'roll_{period}_close'] = df['close']

            roll_range = df[f'roll_{period}_high'] - df[f'roll_{period}_low']
            roll_body = df[f'roll_{period}_close'] - df[f'roll_{period}_open']
            
            df[f'tsi_{period}'] = (roll_body / roll_range).replace([np.inf, -np.inf], 0).fillna(0)

    def get_entry_signal(self, row, portfolio):
        base_signal, base_signal_type = super().get_entry_signal(row, portfolio)
        if base_signal == 0:
            return 0, "none"
        
        if pd.isna(row.atr) or pd.isna(row.historical_avg_atr) or row.historical_avg_atr == 0:
            return 0, "atr_data_missing"
            
        ratio = row.atr / row.historical_avg_atr
        
        if ratio < self.volatility_thresholds['low']:
            regime = 'low'
        elif ratio > self.volatility_thresholds['high']:
            regime = 'high'
        else:
            regime = 'normal'
            
        lookback_periods = self.regime_lookbacks[regime]
        
        confirmation_count = 0
        if base_signal == 1:
            for period in lookback_periods:
                if getattr(row, f'tsi_{period}', 0) >= self.strength_threshold:
                    confirmation_count += 1
        elif base_signal == -1:
            for period in lookback_periods:
                if getattr(row, f'tsi_{period}', 0) <= -self.strength_threshold:
                    confirmation_count += 1

        if confirmation_count >= self.min_confirmation:
            return base_signal, f"{base_signal_type}_atr_bull_{regime}" 
        return 0, f"dynamic_atr_filtered_{regime}"

class Backtester:
    def __init__(self, data: pd.DataFrame, portfolio: Portfolio, strategy: Strategy, params: dict):
        self.data = data
        self.portfolio = portfolio
        self.strategy = strategy
        self.params = params

    def run(self):
        for day, day_data in tqdm(self.data.groupby(self.data.index.date), desc="Backtesting"):
            self.portfolio.record_daily_state(day)
            daily_trade_amount = self.portfolio.history.loc[day, 'aum']
            
            for i, row in enumerate(day_data.itertuples()):
                is_eod = i == len(day_data) - 1
                is_trade_time = (row.min_from_open >= 0 and row.min_from_open % self.params['trade_freq'] == 0)

                if self.portfolio.position['direction'] != 0:
                    action = self.strategy.get_exit_action(row, self.portfolio)
                    if action or is_eod:
                        if action and 'reason' in action:
                            reason = action['reason']
                            if reason == 'ladder_stop':
                                exit_reason = f"L{action['level']} Stop"
                            elif reason == 'ladder_profit':
                                level = action['level']
                                size_pct = int(action.get('size', 1.0) * 100)
                                exit_reason = f"L{level} Profit ({size_pct}%)"
                            elif reason == 'band_reentry':
                                exit_reason = "Band Reentry"
                            elif reason == 'vwap_exit':
                                exit_reason = "VWAP Exit"
                            elif reason == 'atr_stop':
                                exit_reason = "ATR Stop"
                            elif reason == 'atr_profit':
                                exit_reason = "ATR Profit"
                            else:
                                exit_reason = reason
                        elif is_eod:
                            exit_reason = "EOD"
                        else:
                            exit_reason = "Unknown"
                        
                        self.portfolio.execute_exit(row.close, row.Index, action or {'type': 'full'}, exit_reason)
                        if is_eod: continue
                
                if is_trade_time:
                    signal, signal_type = self.strategy.get_entry_signal(row, self.portfolio)
                    if signal != 0:
                        self.portfolio.execute_entry(signal, row.close, row.Index, daily_trade_amount, signal_type)
        
        history = self.portfolio.history.copy()
        history['daily_return'] = history['aum'].pct_change().fillna(0)
        history['daily_gross_return'] = history['gross_aum'].pct_change().fillna(0)
        history['cumulative_return'] = (history['aum'] / self.portfolio.initial_aum) - 1
        history['cumulative_gross_return'] = (history['gross_aum'] / self.portfolio.initial_aum) - 1
        return history, pd.DataFrame(self.portfolio.trades)

class Reporter:
    def __init__(self, daily_results, trades, initial_aum, feature_data=None):
        self.daily_results = daily_results
        self.trades = trades
        self.initial_aum = initial_aum
        self.feature_data = feature_data
        self.strategy_params = {}

    def _calculate_drawdowns(self):
        if self.daily_results.empty:
            return None, None, None, None
            
        cumulative_returns = self.daily_results['aum'] / self.initial_aum
        running_max = cumulative_returns.cummax()
        
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        gross_cumulative_returns = self.daily_results['gross_aum'] / self.initial_aum
        gross_running_max = gross_cumulative_returns.cummax()
        gross_drawdowns = (gross_cumulative_returns - gross_running_max) / gross_running_max
        gross_max_drawdown = gross_drawdowns.min()
        
        return drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown

    def print_summary(self):
        if self.daily_results.empty: print("No results to summarize."); return
        
        # Calculate total returns
        total_net_return = (self.daily_results['aum'].iloc[-1] / self.initial_aum) - 1
        total_gross_return = (self.daily_results['gross_aum'].iloc[-1] / self.initial_aum) - 1
        total_trades = len(self.trades)
        
        # Calculate CAGR
        start_date = self.daily_results.index[0]
        end_date = self.daily_results.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25 if days > 0 else 1
        
        net_cagr = ((self.daily_results['aum'].iloc[-1] / self.initial_aum) ** (1/years) - 1) if years > 0 else 0
        gross_cagr = ((self.daily_results['gross_aum'].iloc[-1] / self.initial_aum) ** (1/years) - 1) if years > 0 else 0
        
        # Calculate maximum drawdowns
        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        
        summary = {
            "Total Net Return": f"{total_net_return:.2%}",
            "Total Gross Return": f"{total_gross_return:.2%}",
            "CAGR (Net)": f"{net_cagr:.2%}",
            "CAGR (Gross)": f"{gross_cagr:.2%}",
            "Maximum Drawdown (Net)": f"{max_drawdown:.2%}" if max_drawdown is not None else "N/A",
            "Maximum Drawdown (Gross)": f"{gross_max_drawdown:.2%}" if gross_max_drawdown is not None else "N/A",
            "Total Trades": total_trades,
        }
        
        if not self.trades.empty:
            # Calculate win rate and average returns
            win_rate = (self.trades['net_pnl'] > 0).mean()
            avg_gross_return = self.trades['gross_return'].mean()
            avg_net_return = self.trades['net_return'].mean()
            
            # Calculate commission metrics
            total_commission = self.trades['commission'].sum()
            avg_commission_per_trade = total_commission / total_trades if total_trades > 0 else 0
            
            # Calculate profit/loss ratio
            winning_trades = self.trades[self.trades['net_return'] > 0]['net_return']
            losing_trades = self.trades[self.trades['net_return'] < 0]['net_return']
            
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                avg_win = winning_trades.mean()
                avg_loss = abs(losing_trades.mean())
                profit_loss_ratio = avg_win / avg_loss
            else:
                profit_loss_ratio = None
        
            signal_breakdown = self.trades['signal_type'].value_counts()
            
            summary.update({
                "Win Rate": f"{win_rate:.2%}",
                "Avg Gross Return/Trade": f"{avg_gross_return:.2%}",
                "Avg Net Return/Trade": f"{avg_net_return:.2%}",
                "Profit/Loss Ratio": f"{profit_loss_ratio:.2f}" if profit_loss_ratio is not None else "N/A",
                "Total Commission": f"{total_commission:,.0f}",
                "Avg Commission/Trade": f"{avg_commission_per_trade:,.0f}",
            })
        else:
            summary["Win Rate"] = "N/A"
        
        print("\n--- Backtest Summary ---")
        [print(f"{k:<25}: {v}") for k, v in summary.items()]
        
        if not self.trades.empty and len(signal_breakdown) > 0:
            print(f"\n--- Signal Type Breakdown ---")
            for signal_type, count in signal_breakdown.items():
                pct = count / total_trades * 100
                avg_return = self.trades[self.trades['signal_type'] == signal_type]['net_return'].mean()
                print(f"{signal_type:<20}: {count:>3} trades ({pct:>5.1f}%) | Avg Return: {avg_return:>6.2%}")
                
        print("-" * 50)

    def plot_aum_curve(self):
        if self.daily_results.empty: return
        
        self.daily_results.index = pd.to_datetime(self.daily_results.index)

        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 18), constrained_layout=True)
        
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.5], hspace=0.2)
        gs_top = gs[0].subgridspec(3, 1, hspace=0)
        
        ax1 = fig.add_subplot(gs_top[0])
        ax2 = fig.add_subplot(gs_top[1], sharex=ax1)
        ax3 = fig.add_subplot(gs_top[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[1])

        ax1.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='x', labelbottom=False)

        self.daily_results['gross_aum'].plot(ax=ax1, label='AUM (Gross, Pre-Cost)', color='green', linestyle='--')
        self.daily_results['aum'].plot(ax=ax1, label='AUM (Net, Post-Cost)', color='blue')
        
        if not self.daily_results.empty and not self.daily_results['aum'].empty:
            start_date, end_date = self.daily_results.index[0], self.daily_results.index[-1]
            total_days = (end_date - start_date).days
            if total_days <= 0:
                total_days = 1

            points_to_annotate = {}
            
            peak_aum_date = self.daily_results['aum'].idxmax()
            points_to_annotate['peak_aum'] = {
                'date': peak_aum_date,
                'value': self.daily_results.loc[peak_aum_date, 'aum'],
                'text': f'Peak AUM\n{self.daily_results.loc[peak_aum_date, "aum"]:,.0f}\n{peak_aum_date.strftime("%Y-%m-%d")}'
            }
            trough_aum_date = self.daily_results['aum'].idxmin()
            points_to_annotate['trough_aum'] = {
                'date': trough_aum_date,
                'value': self.daily_results.loc[trough_aum_date, 'aum'],
                'text': f'Trough AUM\n{self.daily_results.loc[trough_aum_date, "aum"]:,.0f}\n{trough_aum_date.strftime("%Y-%m-%d")}'
            }

            if 'daily_return' in self.daily_results.columns:
                best_return_date = self.daily_results['daily_return'].idxmax()
                best_return_val = self.daily_results.loc[best_return_date, 'daily_return']
                points_to_annotate['best_return'] = {
                    'date': best_return_date,
                    'value': self.daily_results.loc[best_return_date, 'aum'],
                    'text': f'Best Return\n{best_return_val:+.2%}\n{best_return_date.strftime("%Y-%m-%d")}'
                }
                worst_return_date = self.daily_results['daily_return'].idxmin()
                worst_return_val = self.daily_results.loc[worst_return_date, 'daily_return']
                points_to_annotate['worst_return'] = {
                    'date': worst_return_date,
                    'value': self.daily_results.loc[worst_return_date, 'aum'],
                    'text': f'Worst Return\n{worst_return_val:+.2%}\n{worst_return_date.strftime("%Y-%m-%d")}'
                }
            
            annotation_config = {
                'peak_aum':     {'va': 'bottom', 'offset_y': 15,   'color': 'green',  'marker': 'o'},
                'trough_aum':   {'va': 'top',    'offset_y': -15,  'color': 'red',    'marker': 'o'},
                'best_return':  {'va': 'bottom', 'offset_y': 40,   'color': 'blue',   'marker': '*'},
                'worst_return': {'va': 'top',    'offset_y': -40,  'color': 'purple', 'marker': '*'},
            }

            for key, point in points_to_annotate.items():
                config = annotation_config.get(key)
                if not config: continue
                
                is_left_half = (point['date'] - start_date).days < total_days / 2
                ha = 'left' if is_left_half else 'right'
                xytext_x = 50 if is_left_half else -50
                connection_rad = 0.2 if is_left_half else -0.2
                
                plot_date = point['date'].to_pydatetime()

                ax1.scatter(plot_date, point['value'], marker=config['marker'], s=150, facecolors='none', edgecolors=config['color'], linewidths=2.5, zorder=5)
                ax1.annotate(point['text'],
                             xy=(plot_date, point['value']),
                             xytext=(xytext_x, config['offset_y']),
                             textcoords='offset points',
                             ha=ha,
                             va=config['va'],
                             arrowprops=dict(arrowstyle='-|>,' + f'head_width=0.4,' + f'head_length=0.8',
                                             connectionstyle=f'arc3,rad={connection_rad}',
                                             color=config['color'],
                                             lw=1.5),
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=config['color'], lw=1, alpha=0.8))

        price_drawdowns = None
        max_price_dd = None
        
        if self.feature_data is not None and not self.feature_data.empty:
            daily_close = self.feature_data.groupby(self.feature_data.index.date)['close'].last()
            
            daily_results_datetime_index = self.daily_results.index
            daily_close = daily_close[daily_close.index.isin(daily_results_datetime_index.date)]
            
            if not daily_close.empty:
                ax1_twin = ax1.twinx()
                
                initial_price = daily_close.iloc[0]
                close_cumulative_return = (daily_close / initial_price - 1) * 100
                close_cumulative_return.plot(ax=ax1_twin, label='Buy&Hold Return (%)', color='gray', alpha=0.8, linewidth=1.5)
                ax1_twin.set_ylabel('Buy&Hold Cumulative Return (%)', color='gray')
                ax1_twin.tick_params(axis='y', labelcolor='gray')
                
                price_returns = daily_close / initial_price
                price_running_max = price_returns.cummax()
                price_drawdowns = (price_returns - price_running_max)
                max_price_dd = price_drawdowns.min()
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax1_twin.legend().set_visible(False)
        
        ax1.set_title('AUM and Buy&Hold Cumulative Return Over Time', fontsize=16)
        ax1.set_ylabel('AUM')
        if self.feature_data is None or self.feature_data.empty:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if drawdowns is not None:
            gross_drawdowns.plot(ax=ax2, label=f'Gross DD (Max: {gross_max_drawdown:.2%})', color='red', linestyle='--', alpha=0.7)
            drawdowns.plot(ax=ax2, label=f'Net DD (Max: {max_drawdown:.2%})', color='darkred')
            ax2.fill_between(drawdowns.index, drawdowns, 0, alpha=0.3, color='red')
            
            if price_drawdowns is not None:
                ax2_twin = ax2.twinx()
                price_drawdowns.plot(ax=ax2_twin, label=f'Buy&Hold DD (Max: {max_price_dd:.2%})', color='gray', alpha=0.8, linewidth=1.5)
                ax2_twin.set_ylabel('Buy&Hold Drawdown (%)', color='gray')
                ax2_twin.tick_params(axis='y', labelcolor='gray')
                ax2_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
                ax2_twin.legend().set_visible(False)
            
            ax2.set_title('Strategy vs Buy&Hold Drawdown Over Time', fontsize=14)
            ax2.set_ylabel('Strategy Drawdown (%)')
            if price_drawdowns is None:
                ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        if self.feature_data is not None and not self.feature_data.empty and 'open' in self.feature_data.columns:
            daily_agg = self.feature_data.groupby(self.feature_data.index.date).agg(
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last')
            )
            daily_agg.index = pd.to_datetime(daily_agg.index)
            
            intraday_power = ((daily_agg['close'] - daily_agg['open']).abs() / 
                              (daily_agg['high'] - daily_agg['low']).abs())
            intraday_power.replace([np.inf, -np.inf], np.nan, inplace=True)
            intraday_power.fillna(0, inplace=True)
            
            daily_results_datetime_index = self.daily_results.index
            intraday_power = intraday_power[intraday_power.index.isin(daily_results_datetime_index)]
            
            if not intraday_power.empty:
                ax3.bar(intraday_power.index, intraday_power, color='purple', alpha=0.6, width=1.0)
                
                ax3.set_title('Intraday Power Ratio (|Close-Open| / |High-Low|)', fontsize=14)
                ax3.set_ylabel('Ratio')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(bottom=0)
                ax3.set_xlabel('Date')
                ax3.tick_params(axis='x', rotation=45)

                daily_returns = self.daily_results['daily_return'].copy()
                combined_data = pd.DataFrame({'power_ratio': intraday_power, 'daily_return': daily_returns}).dropna()
                combined_data.index = pd.to_datetime(combined_data.index)

                if len(combined_data) >= 5:
                    try:
                        labels = ['Strong Loss', 'Loss', 'Neutral', 'Profit', 'Strong Profit']
                        
                        ranks = combined_data['daily_return'].rank(method='first')
                        combined_data['return_category'] = pd.qcut(ranks, q=5, labels=labels)
                        
                        gb = combined_data.groupby('return_category', observed=True)['daily_return']
                        avg_power_by_return = combined_data.groupby('return_category', observed=True)['power_ratio'].mean()
                        
                        if isinstance(avg_power_by_return.index, pd.CategoricalDtype):
                            avg_power_by_return = avg_power_by_return.reindex(labels)

                        colors = plt.get_cmap('gray_r')(np.linspace(0.1, 0.7, len(labels)))
                        avg_power_by_return.plot(kind='bar', ax=ax4, color=colors, alpha=0.8)
                        
                        bins_min = gb.min()
                        bins_max = gb.max()
                        bin_labels = [f"{bins_min[label]:.2%} to {bins_max[label]:.2%}" for label in avg_power_by_return.index if label in bins_min]

                        for i, patch in enumerate(ax4.patches):
                            if i < len(bin_labels):
                                ax4.text(patch.get_x() + patch.get_width() / 2.,
                                         patch.get_height(),
                                         bin_labels[i],
                                         ha='center',
                                         va='bottom',
                                         fontsize=9,
                                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.6, edgecolor='none'))
                        
                        ax4.set_title('Average Power Ratio by Daily Return Quintile', fontsize=14)
                        ax4.set_ylabel('Average Ratio')
                        ax4.set_xlabel('Return Category')
                        ax4.tick_params(axis='x', rotation=0)
                        ax4.grid(True, alpha=0.3, axis='y')

                    except Exception as e:
                        print(f"An unexpected error occurred while creating return quantiles: {e}")
                else:
                    print("Not enough daily returns to create 5 quantiles for analysis.")

        plt.show()

    def plot_candlestick_performance(self):
        if self.trades.empty or self.feature_data is None:
            print("Not enough data for candlestick performance analysis.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        daily_ohlc = self.feature_data.groupby(self.feature_data.index.date).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last')
        )
        daily_ohlc.index = pd.to_datetime(daily_ohlc.index)

        analyzer = CandlestickAnalyzer()
        daily_ohlc['pattern'] = analyzer.classify(daily_ohlc)
        
        trades_df = self.trades.copy()
        trades_df['entry_day'] = pd.to_datetime(trades_df['entry_time']).dt.normalize()
        
        merged_df = pd.merge(trades_df, daily_ohlc[['pattern']], left_on='entry_day', right_index=True)
        
        if merged_df.empty:
            print("No trades could be matched with candlestick patterns.")
            return
            
        performance = merged_df.groupby('pattern')['net_return'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        if performance.empty:
            print("No performance data to plot for candlestick patterns.")
            return

        korean_patterns = {
            "Doji": "십자형",
            "Bullish Marubozu": "장대양봉",
            "Bearish Marubozu": "장대음봉",
            "Hammer / Hanging Man": "망치형/교수형",
            "Shooting Star": "유성형",
            "Spinning Top": "팽이형",
            "Standard Candle": "보통형"
        }
        performance.rename(index=korean_patterns, inplace=True)

        fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
        
        colors = ['green' if x > 0 else 'red' for x in performance['mean']]
        
        bars = sns.barplot(x=performance.index, y=performance['mean'], ax=ax, palette=colors)
        
        for i, bar in enumerate(bars.patches):
            mean_return = performance['mean'].iloc[i]
            trade_count = performance['count'].iloc[i]
            label = f"{mean_return:.2%}\n({trade_count} trades)"
            
            y_pos = bar.get_height()
            va = 'bottom' if mean_return >= 0 else 'top'
            offset = 0.0005 if mean_return >=0 else -0.0005
            
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset, label,
                    ha='center', va=va,
                    fontsize=10, color='black')
                    
        ax.set_title('Average Net Return by Candlestick Pattern on Entry Day', fontsize=16)
        ax.set_ylabel('Average Net Return (%)')
        ax.set_xlabel('캔들 패턴')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.show()

    def plot_drawdown_analysis(self):
        if self.daily_results.empty:
            print("No data available for drawdown analysis.")
            return
            
        drawdowns, max_drawdown, gross_drawdowns, gross_max_drawdown = self._calculate_drawdowns()
        if drawdowns is None:
            print("Unable to calculate drawdowns.")
            return
            
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        drawdowns.plot(ax=ax1, color='red', alpha=0.8)
        ax1.fill_between(drawdowns.index, drawdowns, 0, alpha=0.3, color='red')
        ax1.set_title(f'Net Drawdown (Max: {max_drawdown:.2%})')
        ax1.set_ylabel('Drawdown')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax1.grid(True, alpha=0.3)
        
        drawdowns.hist(bins=50, ax=ax2, alpha=0.7, color='red')
        ax2.axvline(max_drawdown, color='darkred', linestyle='--', linewidth=2, label=f'Max DD: {max_drawdown:.2%}')
        ax2.set_title('Drawdown Distribution')
        ax2.set_xlabel('Drawdown')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        cumulative_returns = self.daily_results['aum'] / self.initial_aum
        running_max = cumulative_returns.cummax()
        
        (cumulative_returns - 1).plot(ax=ax3, label='Cumulative Return', color='blue')
        (running_max - 1).plot(ax=ax3, label='Running Peak', color='green', linestyle='--')
        ax3.fill_between(cumulative_returns.index, cumulative_returns - 1, running_max - 1, alpha=0.3, color='red')
        ax3.set_title('Underwater Plot')
        ax3.set_ylabel('Return')
        ax3.legend()
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax3.grid(True, alpha=0.3)
        
        rolling_dd = drawdowns.rolling(window=30).min()
        rolling_dd.plot(ax=ax4, color='orange', label='30-Day Rolling Max DD')
        ax4.set_title('Rolling Maximum Drawdown (30 Days)')
        ax4.set_ylabel('Max Drawdown')
        ax4.set_xlabel('Date')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.suptitle(f'Drawdown Analysis - MDD: {max_drawdown:.2%}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

    def plot_intraday(self, target_date: str):
        if self.feature_data is None:
            print("Feature data not available. Please pass feature_data to Reporter.")
            return
            
        try:
            target_date_obj = pd.to_datetime(target_date).date()
        except:
            print(f"Invalid date format: {target_date}. Please use 'YYYY-MM-DD' format.")
            return

        available_days = sorted(self.feature_data['day'].unique())
        
        if target_date_obj not in available_days:
            print(f"Date {target_date} is not a trading day or no data available.")
            before_dates = [d for d in available_days if d < target_date_obj][-5:]
            after_dates = [d for d in available_days if d > target_date_obj][:5]
            print("Nearby trading days:")
            if before_dates: print(f"  Before: {', '.join([str(d) for d in before_dates])}")
            if after_dates: print(f"  After: {', '.join([str(d) for d in after_dates])}")
            return

        day_data = self.feature_data[self.feature_data['day'] == target_date_obj].copy()
        if day_data.empty:
            print(f"No data available for date {target_date}.")
            return

        day_trades = []
        if not self.trades.empty:
            for _, trade in self.trades.iterrows():
                entry_date = pd.to_datetime(trade['entry_time']).date()
                exit_date = pd.to_datetime(trade['exit_time']).date()
                if entry_date == target_date_obj or exit_date == target_date_obj:
                    day_trades.append(trade)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        time_labels = day_data.index.strftime('%H:%M')
        x_pos = range(len(day_data))

        ax.plot(x_pos, day_data['close'], label='Close Price', color='black', linewidth=2.5, zorder=3)
        
        if 'vwap' in day_data.columns and day_data['vwap'].notna().any():
            ax.plot(x_pos, day_data['vwap'], label='VWAP', color='purple', linewidth=2, alpha=0.8, zorder=2)
        
        if 'UB' in day_data.columns and day_data['UB'].notna().any():
            ax.plot(x_pos, day_data['UB'], label='Entry Upper Band', color='red', linestyle='-', alpha=0.8, linewidth=1.5)
            ax.plot(x_pos, day_data['LB'], label='Entry Lower Band', color='blue', linestyle='-', alpha=0.8, linewidth=1.5)
            ax.fill_between(x_pos, day_data['UB'], day_data['LB'], alpha=0.08, color='gray', label='Entry Band Zone')

        if 'exit_UB' in day_data.columns and day_data['exit_UB'].notna().any():
            ax.plot(x_pos, day_data['exit_UB'], label='Exit Upper Band', color='darkred', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.plot(x_pos, day_data['exit_LB'], label='Exit Lower Band', color='darkblue', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.fill_between(x_pos, day_data['exit_UB'], day_data['exit_LB'], alpha=0.05, color='orange', label='Exit Band Zone')

        trade_events = []
        trade_colors = ['green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
        
        day_trades_sorted = sorted(day_trades, key=lambda x: pd.to_datetime(x['entry_time']))
        
        for trade_idx, trade in enumerate(day_trades_sorted):
            color_idx = trade_idx % len(trade_colors)
            base_color = trade_colors[color_idx]
            
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            entry_idx = None
            exit_idx = None
            
            if entry_time.date() == target_date_obj:
                entry_idx = day_data.index.get_indexer([entry_time], method='nearest')[0]
                if entry_idx >= 0:
                    direction = 'long' if trade['direction'] > 0 else 'short'
                    marker = '^' if direction == 'long' else 'v'
                    ax.scatter(entry_idx, trade['entry_price'], color=base_color, marker=marker, s=150, zorder=6, edgecolors='white', linewidth=1)
                    signal_type = trade.get('signal_type', 'unknown')
                    trade_events.append(f"T{trade_idx+1}: {entry_time.strftime('%H:%M')} - {direction.upper()} Entry ({signal_type}): {trade['entry_price']:.0f}")
            
            if exit_time.date() == target_date_obj:
                exit_idx = day_data.index.get_indexer([exit_time], method='nearest')[0]
                if exit_idx >= 0:
                    ax.scatter(exit_idx, trade['exit_price'], color=base_color, marker='x', s=150, zorder=6, linewidth=3)
                    net_return = trade.get('net_return', 0)
                    commission = trade.get('commission', 0)
                    exit_reason = trade.get('exit_reason', 'Unknown')
                    trade_events.append(f"T{trade_idx+1}: {exit_time.strftime('%H:%M')} - Exit ({exit_reason}): {trade['exit_price']:.0f} | Return: {net_return:.2%} | Cost: {commission:.0f}")

            if entry_idx is not None:
                start_x = entry_idx
                end_x = exit_idx if exit_idx is not None else len(x_pos) - 1

                # ATR Stop/Profit Lines
                if 'atr_multiplier_stop' in self.strategy_params:
                    entry_atr = day_data.iloc[entry_idx]['atr']
                    atr_stop = self.strategy_params.get('atr_multiplier_stop', 2.0)
                    atr_profit = self.strategy_params.get('atr_multiplier_profit', 4.0)
                    
                    stop_price = trade['entry_price'] - trade['direction'] * entry_atr * atr_stop
                    profit_price = trade['entry_price'] + trade['direction'] * entry_atr * atr_profit
                    
                    ax.hlines(stop_price, start_x, end_x, colors='darkred', linestyles='-.', alpha=0.9, linewidth=1.5,
                              label=f'ATR Stop ({atr_stop}x)' if trade_idx == 0 else "")
                    ax.text(start_x + 2, stop_price, f'T{trade_idx+1} ATR Stop', fontsize=8, color='darkred',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    ax.hlines(profit_price, start_x, end_x, colors='darkgreen', linestyles='-.', alpha=0.9, linewidth=1.5,
                              label=f'ATR Profit ({atr_profit}x)' if trade_idx == 0 else "")
                    ax.text(start_x + 2, profit_price, f'T{trade_idx+1} ATR Profit', fontsize=8, color='darkgreen',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

                # Ladder Stop/Profit Lines
                if 'ladder_levels' in self.strategy_params:
                    ladder_levels = self.strategy_params.get('ladder_levels', [])
                    entry_price = trade['entry_price']
                    direction = trade['direction']
                    
                    for i, level in enumerate(ladder_levels):
                        stop_price = entry_price * (1 + level['stop'] * direction)
                        profit_price = entry_price * (1 + level['profit'] * direction)
                        
                        ax.hlines(stop_price, start_x, end_x, colors='red', linestyles=':', alpha=0.8, linewidth=2, 
                                 label=f'L{i} Stop ({level["stop"]*100:+.1f}%)' if trade_idx == 0 and i == 0 else "")
                        ax.text(start_x + 2, stop_price, f'T{trade_idx+1}L{i}S', fontsize=8, color='red', 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                        
                        ax.hlines(profit_price, start_x, end_x, colors='green', linestyles=':', alpha=0.8, linewidth=2,
                                 label=f'L{i} Profit ({level["profit"]*100:+.1f}%)' if trade_idx == 0 and i == 0 else "")
                        ax.text(start_x + 2, profit_price, f'T{trade_idx+1}L{i}P({level["size"]*100:.0f}%)', fontsize=8, color='green',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        daily_pnl = None
        daily_return = None
        if target_date_obj in self.daily_results.index:
            daily_pnl = self.daily_results.loc[target_date_obj, 'pnl']
            if 'daily_return' in self.daily_results.columns:
                daily_return = self.daily_results.loc[target_date_obj, 'daily_return']

        title = f'Intraday Chart - {target_date}'
        if daily_pnl is not None:
            title += f' | Daily PnL: {daily_pnl:,.0f}'
            if daily_return is not None:
                title += f' ({daily_return:.2%})'
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        
        tick_interval = max(1, len(day_data) // 15)
        ax.set_xticks(x_pos[::tick_interval])
        ax.set_xticklabels(time_labels[::tick_interval], rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        if trade_events:
            print(f"\n=== {target_date} Trade Summary ===")
            if daily_pnl is not None:
                print(f"Daily PnL: {daily_pnl:,.0f}")
                if daily_return is not None:
                    print(f"Daily Return: {daily_return:.2%}")
            
            print("\nTrade Details:")
            for event in trade_events:
                print(f"  {event}")
                
            entry_trades = [event for event in trade_events if 'Entry' in event]
            exit_trades = [event for event in trade_events if 'Exit' in event]
            
            print(f"\n=== Daily Trading Statistics ===")
            print(f"  Entries: {len(entry_trades)} | Exits: {len(exit_trades)}")
            
            exit_reasons = {}
            for event in exit_trades:
                if '(' in event and ')' in event:
                    reason_start = event.find('Exit (') + 6
                    reason_end = event.find(')', reason_start)
                    if reason_start > 5 and reason_end > reason_start:
                        reason = event[reason_start:reason_end]
                        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            if exit_reasons:
                print(f"  Exit Reasons:")
                for reason, count in exit_reasons.items():
                    print(f"    {reason}: {count}")
            
            day_trade_returns = [trade.get('net_return', 0) for trade in day_trades]
            day_trade_commissions = [trade.get('commission', 0) for trade in day_trades]
            
            if day_trade_returns:
                print(f"\n=== Performance Metrics ===")
                print(f"  Total Completed Trades: {len(day_trades)}")
                print(f"  Winning Trades: {sum(1 for r in day_trade_returns if r > 0)}")
                print(f"  Losing Trades: {sum(1 for r in day_trade_returns if r < 0)}")
                print(f"  Win Rate: {sum(1 for r in day_trade_returns if r > 0) / len(day_trade_returns):.1%}")
                print(f"  Average Return: {np.mean(day_trade_returns):.2%}")
                print(f"  Best Trade: {max(day_trade_returns):.2%}")
                print(f"  Worst Trade: {min(day_trade_returns):.2%}")
                print(f"  Total Commission: {sum(day_trade_commissions):,.0f}")
                
            if hasattr(self, 'ladder_levels') and self.ladder_levels:
                print(f"\n=== Ladder Level Analysis ===")
                for trade_idx, trade in enumerate(day_trades_sorted):
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    
                    if entry_time.date() == target_date_obj:
                        entry_price = trade['entry_price']
                        direction = trade['direction']
                        exit_reason = trade.get('exit_reason', 'Unknown')
                        
                        trade_data = day_data[(day_data.index >= entry_time) & (day_data.index <= exit_time)]
                        if not trade_data.empty:
                            min_price = trade_data['low'].min()
                            max_price = trade_data['high'].max()
                            
                            print(f"\nT{trade_idx+1} ({entry_time.strftime('%H:%M')}-{exit_time.strftime('%H:%M')}) Exit: {exit_reason}")
                            print(f"  Entry: {entry_price:.0f}, Exit: {trade['exit_price']:.0f}")
                            print(f"  Price Range: {min_price:.0f} - {max_price:.0f}")
                            
                            for i, level in enumerate(self.ladder_levels):
                                stop_price = entry_price * (1 + level['stop'] * direction)
                                profit_price = entry_price * (1 + level['profit'] * direction)
                                
                                if direction == 1:  # Long
                                    stop_reached = min_price <= stop_price
                                    profit_reached = max_price >= profit_price
                                else:  # Short
                                    stop_reached = max_price >= stop_price
                                    profit_reached = min_price <= profit_price
                                
                                print(f"  L{i} Stop {stop_price:.0f}: {'✓' if stop_reached else '✗'}")
                                print(f"  L{i} Profit {profit_price:.0f}: {'✓' if profit_reached else '✗'}")
        else:
            print(f"\nNo trades occurred on {target_date}.")
            if daily_pnl is not None:
                print(f"Daily PnL: {daily_pnl:,.0f}")
                if daily_return is not None:
                    print(f"Daily Return: {daily_return:.2%}")


class CandlestickAnalyzer:
    def __init__(self,
                 body_thresh: float = 0.8,
                 wick_ratio_thresh: float = 2.0,
                 doji_thresh: float = 0.1,
                 spinning_top_body_thresh: float = 0.3):
        self.params = {
            'body_thresh': body_thresh,
            'wick_ratio_thresh': wick_ratio_thresh,
            'doji_thresh': doji_thresh,
            'spinning_top_body_thresh': spinning_top_body_thresh,
        }

    def classify(self, ohlc_df: pd.DataFrame) -> pd.Series:
        return ohlc_df.apply(self._get_pattern, axis=1)

    def _get_pattern(self, row: pd.Series) -> str:
        body = abs(row.close - row.open)
        total_range = row.high - row.low

        if total_range < 1e-9:
            return "Doji"

        body_ratio = body / total_range
        upper_wick = row.high - max(row.open, row.close)
        lower_wick = min(row.open, row.close) - row.low

        is_bullish = row.close > row.open

        if body_ratio < self.params['doji_thresh']:
            return "Doji"

        if body_ratio > self.params['body_thresh']:
            return "Bullish Marubozu" if is_bullish else "Bearish Marubozu"
        
        body_not_zero = body > 1e-9
        
        if body_not_zero and lower_wick / body > self.params['wick_ratio_thresh'] and upper_wick / total_range < self.params['doji_thresh']:
            return "Hammer / Hanging Man"

        if body_not_zero and upper_wick / body > self.params['wick_ratio_thresh'] and lower_wick / total_range < self.params['doji_thresh']:
            return "Shooting Star"

        if body_ratio < self.params['spinning_top_body_thresh']:
            return "Spinning Top"
        return "Standard Candle"

class RollingCandleStrategy(BandVWAPStrategy):
    def __init__(self, data, params):
        self.rolling_periods = params.get('rolling_periods', [30, 60, 120])
        self.strength_threshold = params.get('strength_threshold', 0.7)
        self.min_confirmation = params.get('min_confirmation', 2)
        super().__init__(data, params)

    def _calculate_bands(self):
        super()._calculate_bands()
        self._calculate_rolling_candles()

    def _calculate_rolling_candles(self):
        df = self.data
        for period in self.rolling_periods:
            df[f'roll_{period}_open'] = df['open'].shift(period - 1)
            df[f'roll_{period}_high'] = df['high'].rolling(window=period).max()
            df[f'roll_{period}_low'] = df['low'].rolling(window=period).min()
            df[f'roll_{period}_close'] = df['close']

            roll_range = df[f'roll_{period}_high'] - df[f'roll_{period}_low']
            roll_body = df[f'roll_{period}_close'] - df[f'roll_{period}_open']
            
            df[f'tsi_{period}'] = (roll_body / roll_range).replace([np.inf, -np.inf], 0).fillna(0)

    def get_entry_signal(self, row, portfolio):
        base_signal, base_signal_type = super().get_entry_signal(row, portfolio)

        if base_signal == 0:
            return 0, "none"
            
        confirmation_count = 0
        if base_signal == 1:
            for period in self.rolling_periods:
                if getattr(row, f'tsi_{period}', 0) >= self.strength_threshold:
                    confirmation_count += 1
        elif base_signal == -1:
            for period in self.rolling_periods:
                if getattr(row, f'tsi_{period}', 0) <= -self.strength_threshold:
                    confirmation_count += 1
        
        if confirmation_count >= self.min_confirmation:
            signal_type = f"{base_signal_type}_roll_bull" if base_signal == 1 else f"{base_signal_type}_roll_bear"
            return base_signal, signal_type
        else:
            return 0, "rolling_filtered"


def run_backtest(config):
    loader = Loader(config['asset_code'], config['data_folder'])

    fe_params = {
        'loader': loader,
        'start_date': config['start_date'],
        'end_date': config['end_date'],
        'rolling_move': config['rolling_move'],
    }
    if 'atr_period' in config.get('strategy_params', {}):
        fe_params['atr_period'] = config['strategy_params']['atr_period']
    if 'historical_atr_days' in config:
        fe_params['historical_atr_days'] = config['historical_atr_days']
    
    feature_engineer = FeatureEngineer(**fe_params)
    feature_data = feature_engineer.create_features()

    if not feature_data.empty:
        portfolio = Portfolio(config['initial_aum'], config['commission_rate'], config['tax_rate'])
        strategy = config['strategy_class'](feature_data, config['strategy_params'])
        backtester = Backtester(feature_data, portfolio, strategy, config)
        
        daily_results, trades_df = backtester.run()

        reporter = Reporter(daily_results, trades_df, config['initial_aum'], feature_data)
        reporter.strategy_params = config.get('strategy_params', {})
        reporter.print_summary()
        reporter.plot_aum_curve()
        reporter.plot_candlestick_performance()
        
        return reporter, daily_results, trades_df
    else:
        return None, None

if __name__ == "__main__":
    
    config_original = {
        "name": "Original Band-VWAP Strategy",
        "strategy_class": BandVWAPStrategy,
        "asset_code": "A233740_merged", 
        "data_folder": "DATA",
        "start_date": "2020-01-01", 
        "end_date": "2025-07-09",
        "rolling_move": 14,
        "trade_freq": 15,
        "initial_aum": 1e8,
        "commission_rate": 0.02 / 100, 
        "tax_rate": 0.0,
        "strategy_params": {
            "band_multiplier": 1.0,
            "use_vwap": True,
        }
    }

    config_improved = {
        "name": "Improved Ladder-Exit Strategy",
        "strategy_class": LadderExitStrategy,
        "asset_code": "A233740_merged", 
        "data_folder": "DATA",
        "start_date": "2020-01-01", 
        "end_date": "2025-07-09",
        "rolling_move": 14,
        "trade_freq": 5,
        "initial_aum": 1e8,
        "commission_rate": 0.02 / 100, 
        "tax_rate": 0.0,
        "strategy_params": {
            "band_multiplier": 1.5,
            "exit_band_multiplier": 1.0,
            "use_vwap": True,
            "ladder_levels": [
                {'stop': -0.015, 'profit': 0.025, 'size': 0.5},
                {'stop': 0.015, 'profit': 0.04, 'size': 1.0},
            ]
        }
    }

    config_ma = {
        "name": "MA-Filtered Ladder-Exit Strategy",
        "strategy_class": LadderExitMAStrategy,
        "asset_code": "A102110", 
        "data_folder": "DATA",
        "start_date": "2020-01-01", 
        "end_date": "2024-06-20",
        "rolling_move": 14,
        "trade_freq": 15,
        "initial_aum": 1e8,
        "commission_rate": 0.02 / 100, 
        "tax_rate": 0.0,
        "strategy_params": {
            "band_multiplier": 1.29,
            "exit_band_multiplier": 0.78,
            "use_vwap": True,
            "ma_period": 120,
            "ladder_levels": [
                {'stop': -0.01, 'profit': 0.02, 'size': 0.5},
                {'stop': 0.01, 'profit': 0.05, 'size': 1.0},
            ]
        }
    }

    config_atr = {
        "name": "ATR Stop Strategy",
        "strategy_class": ATRStopStrategy,
        "asset_code": "A233740_merged",
        "data_folder": "DATA",
        "start_date": "2020-01-01",
        "end_date": "2025-07-09",
        "rolling_move": 14,
        "trade_freq": 15,
        "initial_aum": 1e8,
        "commission_rate": 0.02 / 100,
        "tax_rate": 0.0,
        "strategy_params": {
            "band_multiplier": 1.5,
            "use_vwap": True,
            "atr_period": 14,
            "atr_multiplier_stop": 1,
            "atr_multiplier_profit": 2
        }
    }

    config_rolling = {
        "name": "Rolling Candle Multi-Timeframe Strategy",
        "strategy_class": RollingCandleStrategy,
        "asset_code": "A233740_merged",
        "data_folder": "DATA",
        "start_date": "2020-01-01",
        "end_date": "2025-07-09",
        "rolling_move": 14,
        "trade_freq": 15,
        "initial_aum": 1e8,
        "commission_rate": 0.02 / 100,
        "tax_rate": 0.0,
        "strategy_params": {
            "band_multiplier": 1.5,
            "use_vwap": True,
            "rolling_periods": [30, 60, 120],
            "strength_threshold": 0.7,
            "min_confirmation": 2,
        }
    }

    config_atr_dynamic = {
        "name": "Dynamic Multi-Candle ATR Strategy",
        "strategy_class": RollingCandleATRStrategy,
        "asset_code": "A233740_merged",
        "data_folder": "DATA",
        "start_date": "2020-01-01",
        "end_date": "2025-07-09",
        "rolling_move": 14,
        "trade_freq": 15,
        "historical_atr_days": 20,
        "initial_aum": 1e8,
        "commission_rate": 0.02 / 100,
        "tax_rate": 0.0,
        "strategy_params": {
            "band_multiplier": 1.5,
            "use_vwap": True,
            "volatility_thresholds": {'low': 0.7, 'high': 1.5},
            "regime_lookbacks": {
                'low': [120, 180, 240],
                'normal': [60, 120, 180],
                'high': [15, 30, 60]
            },
            "strength_threshold": 0.7,
            "min_confirmation": 2
        }
    }

    reporter, daily_results, trades_df = run_backtest(config_improved)
    if reporter:
        reporter.plot_intraday('2023-11-06')