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
import numpy as np
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

# %%
df, metadata = load_data()

# %% [markdown]
# # Prediction using top 20 most viewed content items

# %%
train, test = sample_users(df, seed=117)

# %% [markdown]
# ## Find top content for train users

# %%
top_content = train[train.tunein.dt.month != 3].content_id.value_counts()

# %% [markdown]
# ## Make predictions for test users on M12

# %%
# M12 for test
df_test_m12 = pd.DataFrame(index=test.account_id.unique())

# Add seen on M12
df_test_m12 = df_test_m12.join(
    test[test.tunein.dt.month != 3].groupby("account_id")[["content_id"]].agg(set).rename(columns={"content_id": "seen_m12"}),
    how="left"
)

# Fill those who didn't watch anything
df_test_m12.loc[df_test_m12.seen_m12.isna(), "seen_m12"] = set()

def assign_top20_unseen(seen):
    return top_content[~top_content.index.isin(set(seen))].head(20).index.tolist()

# Make predictions
df_test_m12["preds"] = df_test_m12.seen_m12.apply(assign_top20_unseen)

# %% [markdown]
# ## Evaluate over M3 of test users

# %%
# M3 for test
df_test_m3 = pd.DataFrame(index=test.account_id.unique())

# Add seen on M3
df_test_m3 = df_test_m3.join(
    test[test.tunein.dt.month == 3].groupby("account_id")[["content_id"]].agg(set).rename(columns={"content_id": "seen_m3"}),
    how="left"
)

# Fill those who didn't watch anything
df_test_m3.loc[df_test_m3.seen_m3.isna(), "seen_m3"] = set()

# Required for MAP
df_test_m3["seen_m3"] = df_test_m3["seen_m3"].apply(list)

# %%
merged = pd.merge(df_test_m3, df_test_m12, left_index=True, right_index=True, how="outer")

assert not merged.isna().any().any()

# %%
MAP(
    merged["seen_m3"].tolist(),
    merged["preds"].tolist()
)

# %% [markdown]
# # Prediction using top 20 most viewed content items normalized by assets qty

# %%
train, test = sample_users(df, seed=117)

# %% [markdown]
# ## Find top content for train users

# %%
most_viewed_content = train[train.tunein.dt.month != 3].content_id.value_counts().sort_index()

# %%
assets_per_content = train[train.tunein.dt.month != 3].groupby("content_id").asset_id.agg("nunique").sort_index()

# %%
top_content = (most_viewed_content / np.log(assets_per_content + 1)).sort_values(ascending=False)

# %% [markdown]
# ## Make predictions for test users on M12

# %%
# M12 for test
df_test_m12 = pd.DataFrame(index=test.account_id.unique())

# Add seen on M12
df_test_m12 = df_test_m12.join(
    test[test.tunein.dt.month != 3].groupby("account_id")[["content_id"]].agg(set).rename(columns={"content_id": "seen_m12"}),
    how="left"
)

# Fill those who didn't watch anything
df_test_m12.loc[df_test_m12.seen_m12.isna(), "seen_m12"] = set()

def assign_top20_unseen(seen):
    return top_content[~top_content.index.isin(set(seen))].head(20).index.tolist()

# Make predictions
df_test_m12["preds"] = df_test_m12.seen_m12.apply(assign_top20_unseen)

# %% [markdown]
# ## Evaluate over M3 of test users

# %%
# M3 for test
df_test_m3 = pd.DataFrame(index=test.account_id.unique())

# Add seen on M3
df_test_m3 = df_test_m3.join(
    test[test.tunein.dt.month == 3].groupby("account_id")[["content_id"]].agg(set).rename(columns={"content_id": "seen_m3"}),
    how="left"
)

# Fill those who didn't watch anything
df_test_m3.loc[df_test_m3.seen_m3.isna(), "seen_m3"] = set()

# Required for MAP
df_test_m3["seen_m3"] = df_test_m3["seen_m3"].apply(list)

# %%
merged = pd.merge(df_test_m3, df_test_m12, left_index=True, right_index=True, how="outer")

assert not merged.isna().any().any()

# %%
MAP(
    merged["seen_m3"].tolist(),
    merged["preds"].tolist()
)

# %% [markdown]
# # Prediction using top 20 most viewed content items dropping resumes

# %%
train, test = sample_users(df, seed=117)

# %%
print(len(train))
train.drop_duplicates(subset=["account_id", "content_id"], inplace=True)
print(len(train))

# %% [markdown]
# ## Find top content for train users

# %%
top_content = train[train.tunein.dt.month != 3].content_id.value_counts()

# %% [markdown]
# ## Make predictions for test users on M12

# %%
# M12 for test
df_test_m12 = pd.DataFrame(index=test.account_id.unique())

# Add seen on M12
df_test_m12 = df_test_m12.join(
    test[test.tunein.dt.month != 3].groupby("account_id")[["content_id"]].agg(set).rename(columns={"content_id": "seen_m12"}),
    how="left"
)

# Fill those who didn't watch anything
df_test_m12.loc[df_test_m12.seen_m12.isna(), "seen_m12"] = set()

def assign_top20_unseen(seen):
    return top_content[~top_content.index.isin(set(seen))].head(20).index.tolist()

# Make predictions
df_test_m12["preds"] = df_test_m12.seen_m12.apply(assign_top20_unseen)

# %% [markdown]
# ## Evaluate over M3 of test users

# %%
# M3 for test
df_test_m3 = pd.DataFrame(index=test.account_id.unique())

# Add seen on M3
df_test_m3 = df_test_m3.join(
    test[test.tunein.dt.month == 3].groupby("account_id")[["content_id"]].agg(set).rename(columns={"content_id": "seen_m3"}),
    how="left"
)

# Fill those who didn't watch anything
df_test_m3.loc[df_test_m3.seen_m3.isna(), "seen_m3"] = set()

# Required for MAP
df_test_m3["seen_m3"] = df_test_m3["seen_m3"].apply(list)

# %%
merged = pd.merge(df_test_m3, df_test_m12, left_index=True, right_index=True, how="outer")

assert not merged.isna().any().any()

# %%
MAP(
    merged["seen_m3"].tolist(),
    merged["preds"].tolist()
)

# %%
