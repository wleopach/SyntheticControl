import pickle
from utils import *
import causalpy as cp
import matplotlib.pyplot as plt
import arviz as az
seed = 42
with open('Data/sc_time_series.pickle', 'rb') as file:
    ts_dict = pickle.load(file)

sel = {f"M{i}" for i in range(32, 43)}
for key in ts_dict:
    ts_dict[key] = {seller: ts_dict[key][seller] for seller in sel}
df_dict = {key: df_gen(value) for key, value in ts_dict.items()}
given_date = pd.to_datetime('2023-10-25')


mode = 'ds'
use_cols = {'M36': sel-{'M40'}, 'M40': sel-{'M36'}}
for key in use_cols:
    for kind in df_dict.keys():
        df = df_dict[kind]
        if mode in {'sd', 'raw'}:

            treatment_time = given_date

        else:
            treatment_time = df.index.get_loc(given_date) - 7

        df = df[list(use_cols[key])]
        # starting_date = min(df.index)
        #
        # # Generate a PeriodIndex from the given starting date
        # period_index = pd.period_range(start=starting_date, periods=len(df), freq='D')
        # df.index = period_index
        if mode != 'raw':
            df = gen_tren_df(df, mode)
        result = cp.pymc_experiments.SyntheticControl(
            df,
            treatment_time,
            formula=f"{key} ~ 0 {''.join([f'+{seller}' for seller in sel if seller not in {'M36', 'M40'}])}",
            model=cp.pymc_models.WeightedSumFitter(
                sample_kwargs={"target_accept": 0.95, "random_seed": seed}
            ),
        )

        fig, ax = result.plot(plot_predictors=True)
        plt.savefig(f'Images/{key}-{kind}-{mode}.png')
        print(f'{key}-{kind}')
        result.summary()
        print(az.summary(result.post_impact.mean("obs_ind")))