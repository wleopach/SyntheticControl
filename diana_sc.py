import pickle
from utils import *
import causalpy as cp
import matplotlib.pyplot as plt
seed = 42
with open('Data/sc_time_series.pickle', 'rb') as file:
    ts_dict = pickle.load(file)

sel = {f"M{i}" for i in range(32, 43)}
for key in ts_dict:
    ts_dict[key] = {seller: ts_dict[key][seller] for seller in sel}
df_dict = {key: df_gen(value) for key, value in ts_dict.items()}

df = df_dict['skus_count']
use_cols = sel-{'M40'}

df = df[list(use_cols)]
treatment_time = pd.Timestamp('2023-10-25')
result = cp.pymc_experiments.SyntheticControl(
    df,
    treatment_time,
    formula=f"M36 ~ 0 {''.join([f'+{seller}' for seller in sel if seller not in {'M36', 'M40'}])}",
    model=cp.pymc_models.WeightedSumFitter(
        sample_kwargs={"target_accept": 0.95, "random_seed": seed}
    ),
)

fig, ax = result.plot(plot_predictors=True)
plt.show()
