import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def df_gen(series_dict):
    # Create a DataFrame by concatenating the Series
    df = pd.concat(series_dict, axis=1, keys=series_dict.keys(), sort=True)
    # Fill missing values with 0
    df = df.fillna(0)
    # Convert index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    # Return resulting DataFrame
    return df


def double_exp_smooth_damped(d, extra_periods=1, alpha=0.4, beta=0.4, phi=0.9):
    cols = len(d)  # Historical period length

    d = np.append(d, [np.nan] * extra_periods)  # Append np.nan into the demand array to cover future periods

    # Creation of the level, trend, and forecast arrays
    f, a, b = np.full((3, cols + extra_periods), np.nan)
    # Level & Trend initialization

    a[0] = d[0]

    b[0] = d[1] - d[0]

    # Create all the t+1 forecast
    for t in range(1, cols):
        f[t] = a[t - 1] + phi * b[t - 1]
        a[t] = alpha * d[t] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]

    # Forecast for all extra periods
    for t in range(cols, cols + extra_periods):
        f[t] = a[t - 1] + phi * b[t - 1]
        a[t] = f[t]
        b[t] = phi + b[t - 1]
    df = pd.DataFrame.from_dict({'Demand': d, 'Forecast': f, 'Level': a, 'Trend': b, 'Error': d - f})
    return df


def gen_tren_df(df, mode='sd'):
    new_df = pd.DataFrame()

    for col in df.columns:
        if mode == 'ds':
            new_df[col] = double_exp_smooth_damped(df[col])['Trend']
        else:
            new_df[col] = seasonal_decompose(df[col], period=1, model='additive').trend
    return new_df
