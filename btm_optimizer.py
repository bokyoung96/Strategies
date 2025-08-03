from itertools import product
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

from W_IntraTrading.btm_backtest import run_backtest, LadderExitStrategy

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    if daily_returns.std() == 0:
        return 0
    excess_returns = daily_returns - risk_free_rate / 252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def worker(args):
    config, param_set = args
    
    current_config = copy.deepcopy(config)
    
    current_config['rolling_move'] = param_set['rolling_move']
    current_config['trade_freq'] = param_set['trade_freq']
    current_config['strategy_params']['band_multiplier'] = param_set['band_multiplier']
    current_config['strategy_params']['exit_band_multiplier'] = param_set['exit_band_multiplier']
    current_config['strategy_params']['ladder_levels'][0]['stop'] = param_set['ladder_stop_1']
    current_config['strategy_params']['ladder_levels'][0]['profit'] = param_set['ladder_profit_1']
    current_config['strategy_params']['ladder_levels'][1]['stop'] = param_set['ladder_stop_2']
    current_config['strategy_params']['ladder_levels'][1]['profit'] = param_set['ladder_profit_2']

    daily_results, _ = run_backtest(current_config)
    
    if daily_results is not None and not daily_results.empty:
        sharpe = calculate_sharpe_ratio(daily_results['daily_return'])
        return param_set, sharpe
    return param_set, -np.inf

def main():
    base_config = {
        "name": "Optimization - Ladder-Exit",
        "strategy_class": LadderExitStrategy,
        "asset_code": "A233740_merged",
        "data_folder": "DATA",
        "train_start_date": "2020-01-01",
        "train_end_date": "2023-12-31",
        "test_start_date": "2024-01-01",
        "test_end_date": "2025-07-09",
        "initial_aum": 1e8,
        "commission_rate": 0.02 / 100,
        "tax_rate": 0.0,
        "strategy_params": {
            "use_vwap": True,
            "ladder_levels": [
                {'size': 0.5},
                {'size': 1.0},
            ]
        }
    }

    param_grid = {
        'rolling_move': [5, 14, 20],
        'trade_freq': [5, 10, 15],
        'band_multiplier': [1.0, 1.3, 1.5],
        'exit_band_multiplier': [0.7, 1.0, 1.2],
        'ladder_stop_1': [-0.01, -0.015],
        'ladder_profit_1': [0.02, 0.025],
        'ladder_stop_2': [0.01, 0.005],
        'ladder_profit_2': [0.04, 0.05],
    }

    params_to_test = list(product(*param_grid.values()))
    param_sets = [dict(zip(param_grid.keys(), p)) for p in params_to_test]
    
    print(f"--- Starting Optimization: {len(param_sets)} combinations to test ---")

    train_config = copy.deepcopy(base_config)
    train_config['start_date'] = base_config['train_start_date']
    train_config['end_date'] = base_config['train_end_date']

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap(worker, [(train_config, p) for p in param_sets]), total=len(param_sets)))
        
    best_params, best_sharpe = max(results, key=lambda item: item[1])

    print("\n--- Optimization Complete ---")
    print(f"Best Sharpe Ratio (Train): {best_sharpe:.4f}")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    print("\n--- Running Backtest on Test Set with Best Parameters ---")
    test_config = copy.deepcopy(base_config)
    test_config['start_date'] = base_config['test_start_date']
    test_config['end_date'] = base_config['test_end_date']
    test_config['rolling_move'] = best_params['rolling_move']
    test_config['trade_freq'] = best_params['trade_freq']
    test_config['strategy_params']['band_multiplier'] = best_params['band_multiplier']
    test_config['strategy_params']['exit_band_multiplier'] = best_params['exit_band_multiplier']
    test_config['strategy_params']['ladder_levels'][0]['stop'] = best_params['ladder_stop_1']
    test_config['strategy_params']['ladder_levels'][0]['profit'] = best_params['ladder_profit_1']
    test_config['strategy_params']['ladder_levels'][1]['stop'] = best_params['ladder_stop_2']
    test_config['strategy_params']['ladder_levels'][1]['profit'] = best_params['ladder_profit_2']
    
    run_backtest(test_config)

if __name__ == "__main__":
    main() 