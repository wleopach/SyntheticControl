import pandas as pd


def df_gen(series_dict):
    # Create a DataFrame by concatenating the Series
    df = pd.concat(series_dict, axis=1, keys=series_dict.keys(), sort=True)
    # Fill missing values with 0
    df = df.fillna(0)
    # Convert index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    # Return resulting DataFrame
    return df
