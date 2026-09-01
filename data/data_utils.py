"""Utility functions for our data"""

import pandas as pd


def save_datasets_dict(datasets_dict, data_dir):
    """Saves datasets dictionaries as individual dataframes"""
    for event_time, df in datasets_dict.items():
        file_name = event_time.strftime("%Y%m%d_%H%M%S") + ".parquet"
        df.reset_index().rename(columns={"index": "timestamp"}).to_parquet(
            data_dir / file_name
        )


def load_datasets_into_dict(data_dir):
    """Loads individual dataframes into a dictionary of datasets"""
    out = {}
    for file in sorted(data_dir.glob("*.parquet")):
        event_time = pd.to_datetime(file.stem, format="%Y%m%d_%H%M%S", utc=True)
        out[event_time] = pd.read_parquet(file).set_index("timestamp")
    return out
