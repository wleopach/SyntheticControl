import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import causalpy as cp

seed = 42
df = cp.load_data("sc")
use_cols = ['actual', 'a', 'b', 'c', 'd', 'e','f', 'g']
df = df[use_cols]
treatment_time = 70

# Note, we do not want an intercept in this model
result = cp.pymc_experiments.SyntheticControl(
    df,
    treatment_time,
    formula="actual ~ 0 + a + b + c + d + e + f + g",
    model=cp.pymc_models.WeightedSumFitter(
        sample_kwargs={"target_accept": 0.95, "random_seed": seed}
    ),
)

fig, ax = result.plot(plot_predictors=True)
plt.show()