import pickle
from utils import *
import causalpy as cp
import matplotlib.pyplot as plt
import arviz as az
from data_reader import ts_dict
import argparse

seed = 42

sel = {f"M{i}" for i in range(32, 43)}
for key in ts_dict:
    ts_dict[key] = {seller: ts_dict[key][seller] for seller in sel}
df_dict = {key: df_gen(value) for key, value in ts_dict.items()}
given_date = pd.to_datetime('2023-10-25')
use_cols = {'M36': sel - {'M40'}, 'M40': sel - {'M36'}}


def run(mode='sd'):
    for key in use_cols:
        for kind in df_dict.keys():
            df = df_dict[kind]
            treatment_time = given_date

            df = df[list(use_cols[key])]
            target_column = key

            # Calculate correlation matrix
            correlation_matrix = df[df.index <= treatment_time].corr()

            # Select the top 5 correlated columns with the target column
            top_correlated_columns = correlation_matrix[target_column].abs().sort_values(ascending=False).head(7).index[
                                     1:]

            # Extract the corresponding data from the DataFrame
            top_correlated_columns = list(top_correlated_columns)
            top_correlated_columns.append(key)
            selected_columns_df = df[top_correlated_columns]
            if mode == 'ds':
                treatment_time = df.index.get_loc(given_date) - 7

            # starting_date = min(df.index)
            #
            # # Generate a PeriodIndex from the given starting date
            # period_index = pd.period_range(start=starting_date, periods=len(df), freq='D')
            # df.index = period_index
            if mode != 'raw':
                df_ = gen_tren_df(selected_columns_df, mode)
            else:
                df_ = selected_columns_df
            result = cp.pymc_experiments.SyntheticControl(
                df_,
                treatment_time,
                formula=f"{key} ~ 0 {''.join([f'+{seller}' for seller in top_correlated_columns if seller not in {'M36', 'M40'}])}",
                model=cp.pymc_models.WeightedSumFitter(
                    sample_kwargs={"target_accept": 0.95, "random_seed": seed}
                ),
            )

            fig, ax = result.plot(plot_predictors=True)
            plt.savefig(f'Images/{key}-{kind}-{mode}-top7-cut.png')
            print(f'{key}-{kind}')
            result.summary()
            print(az.summary(result.post_impact.mean("obs_ind")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='sd', type=str)
    args = parser.parse_args()
    run(args.mode)