import os
import glob
import sqlite3
import pandas as pd

class DataValidationError(Exception):
    pass

class DBGetter:
    def __init__(self, 
                 data_folder: str = "DATA", 
                 db_name_base: str = "stock_price(1min)",
                 table_name: str = "A229200"):
        self.folder_path = os.path.join(os.path.dirname(__file__), data_folder)
        self.db_paths = sorted(glob.glob(os.path.join(self.folder_path, f"{db_name_base}_*.db")))
        if not self.db_paths:
            raise FileNotFoundError(f"No database files found matching '{db_name_base}_*.db' in '{self.folder_path}'")
        
        self.table_name = table_name
        # NOTE: After saving data, the dataframes are cached in the instance variables
        self._intra_df: pd.DataFrame | None = None
        self._daily_df: pd.DataFrame | None = None

    def get_intra(self) -> pd.DataFrame:
        dfs = []
        for db_path in self.db_paths:
            try:
                with sqlite3.connect(db_path) as con:
                    df = pd.read_sql(f'SELECT * FROM "{self.table_name}"', con)
                    dfs.append(df)
            except pd.io.sql.DatabaseError:
                print(f"Info: Could not load table '{self.table_name}' from '{os.path.basename(db_path)}'.")
            except Exception as e:
                print(f"An unexpected error occurred with {db_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        res = pd.concat(dfs, ignore_index=True)
        
        if res.duplicated(subset=['date']).any():
            duplicate_count = res.duplicated(subset=['date']).sum()
            print(f"Warning: Found and removed {duplicate_count} duplicate date entries.")
            res = res.drop_duplicates(subset=['date'], keep='first')

        res['date'] = pd.to_datetime(res['date'].astype(str), format='%Y%m%d%H%M')
        res = res.sort_values(by='date').reset_index(drop=True)
        return res

    def get_daily(self, intra_df: pd.DataFrame) -> pd.DataFrame:
        df = intra_df.set_index('caldt')
        
        res = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        })

        close_at_1530 = df.at_time('15:30')['close']
        res['close'] = close_at_1530.resample('D').first()
        
        return res.dropna(how='any').rename_axis('caldt')

    @property
    def intra_df(self) -> pd.DataFrame:
        return self._intra_df

    @property
    def daily_df(self) -> pd.DataFrame:
        return self._daily_df

    def save_data(self):
        intra_df = self.get_intra()
        if intra_df.empty:
            print(f"No data found for ticker {self.table_name}. Nothing to save.")
            return

        intra_df = intra_df.rename(columns={"date": "caldt"})
        daily_df = self.get_daily(intra_df)
        
        intra_unique_days = set(intra_df['caldt'].dt.normalize())
        daily_unique_days = set(daily_df.index)

        if intra_unique_days != daily_unique_days:
            missing_in_daily = sorted(list(intra_unique_days - daily_unique_days))
            extra_in_daily = sorted(list(daily_unique_days - intra_unique_days))
            
            error_parts = []
            if missing_in_daily:
                error_parts.append(f"Missing in daily: {[d.strftime('%Y-%m-%d') for d in missing_in_daily]}")
            if extra_in_daily:
                error_parts.append(f"Extra in daily: {[d.strftime('%Y-%m-%d') for d in extra_in_daily]}")
            raise DataValidationError(f"Date mismatch. {'. '.join(error_parts)}")

        intra_output_path = os.path.join(self.folder_path, f"{self.table_name}_intra.parquet")
        intra_df.to_parquet(intra_output_path, index=False)
        self._intra_df = intra_df.copy()
        print(f"Saved intraday data to '{intra_output_path}'")
        
        daily_output_path = os.path.join(self.folder_path, f"{self.table_name}_daily.parquet")
        daily_df.reset_index(drop=False).to_parquet(daily_output_path, index=False)
        self._daily_df = daily_df.copy()
        print(f"Saved daily summary data to '{daily_output_path}'")


if __name__ == '__main__':
    try:
        db_getter = DBGetter(table_name="A226980")
        db_getter.save_data()
    except (FileNotFoundError, DataValidationError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected system error occurred: {e}")
