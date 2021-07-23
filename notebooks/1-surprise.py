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

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# %%
df, metadata = load_data()

# %%
trainset, val = sample_users(df, seed=117)
data = pd.concat([trainset, val])

trainset = data[data.tunein.dt.month != 3]
val = data[data.tunein.dt.month == 3]

# %%
trainset.head()

# %%
top = trainset.content_id.value_counts(normalize=True).sort_values(ascending=False).cumsum()

plt.figure(figsize=(10,4), dpi=150)
sns.lineplot(np.arange(top.size), top)
plt.axhline(0.9, linestyle=":", color="r")
plt.axhline(0.95, linestyle=":", color="orange")
plt.axhline(0.99, linestyle=":", color="green")

qty90 = (top < 0.90).sum()
qty95 = (top < 0.95).sum()
qty99 = (top < 0.99).sum()

plt.axvline(qty90, linestyle=":", color="r", label=f"90: {qty90} {qty90/len(top):.2%}")
plt.axvline(qty95, linestyle=":", color="orange", label=f"95: {qty95} {qty95/len(top):.2%}")
plt.axvline(qty99, linestyle=":", color="green", label=f"99: {qty99} {qty99/len(top):.2%}")

plt.legend()
plt.show()

# %%
trainset = trainset[trainset.content_id.isin(top[top < 0.99].index)]

# %%
trainset.groupby("account_id").content_id.agg("count").quantile([0.5, 0.9, 0.95, 0.99, 1.0])

# %%
from surprise import NMF, Dataset, Reader
from surprise.model_selection import cross_validate

# %%
trainset = trainset[["account_id", "content_id"]]
trainset["rating"] = 1
trainset.head()

# %%
reader = Reader(rating_scale=(0, 1))
strain = Dataset.load_from_df(trainset, reader)

m = NMF()
cross_validate(m, strain, cv=2)

# %%
valpos = val[["account_id", "content_id"]]
valpos["score"] = 1

# %%
from progressbar import progressbar

# %%
for aid in val.account_id.unique():
    for cid in val.content_id.unique():
        print(m.test([[aid, cid, 0]]))

# %%
m.test(valpos.values)

# %%

# %%
reader = Reader(rating_scale=(0, 1))

# %%
data = Dataset.load_from_df(all_watches, reader)

# %%
strainset = data.build_full_trainset()
stestset = strainset.build_anti_testset()

# %%
algo = NMF()
algo.fit(strainset)

# %%

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

# %%
