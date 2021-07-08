# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from utils import sample_users, MAP, load_data

# %%
df, metadata = load_data()

# %% [markdown]
# # Prediction using top 20 most viewed content items

# %%
train, test = sample_users(df)

# %% [markdown]
# ## Use data of first 2 months

# %%
top20content = train[train.tunein.dt.month != 3].content_id.value_counts()[:20].index.tolist()

# %% [markdown]
# ## Evaluate over M3 of test users

# %%
MAP(
    test[test.tunein.dt.month == 3].groupby("account_id")[["content_id"]].agg(list).content_id.tolist(),
    [top20content] * test[test.tunein.dt.month == 3].account_id.nunique()
)

# %% [markdown]
# # Prediction using top 20 most viewed normalized by qty of assets

# %%
train, test = sample_users(df)

# %% [markdown]
# ## Use data of first 2 months

# %%
content_repetition = train[train.tunein.dt.month != 3].content_id.value_counts().sort_index()
assets_per_content = train[train.tunein.dt.month != 3].groupby("content_id").asset_id.agg("nunique").sort_index()
top20content = (content_repetition / assets_per_content).sort_values(ascending=False).head(20).index

# %% [markdown]
# ## Evaluate over M3 of test users

# %%
MAP(
    test[test.tunein.dt.month == 3].groupby("account_id")[["content_id"]].agg(list).content_id.tolist(),
    [top20content] * test[test.tunein.dt.month == 3].account_id.nunique()
)
