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
treatment_time = pd.Timestamp('2023-10-25')

use_cols = {'M36': sel-{'M40'}, 'M40': sel-{'M36'}}
for key in use_cols:
    for kind in df_dict.keys():
        df = df_dict[kind]
        df = df[list(use_cols[key])]
        result = cp.pymc_experiments.SyntheticControl(
            df,
            treatment_time,
            formula=f"{key} ~ 0 {''.join([f'+{seller}' for seller in sel if seller not in {'M36', 'M40'}])}",
            model=cp.pymc_models.WeightedSumFitter(
                sample_kwargs={"target_accept": 0.95, "random_seed": seed}
            ),
        )

        fig, ax = result.plot(plot_predictors=True)
        plt.savefig(f'Images/{key}-{kind}.png')
        print(f'{key}-{kind}')
        result.summary()
        print(az.summary(result.post_impact.mean("obs_ind")))