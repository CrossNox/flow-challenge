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
def MAP_1M(m=1):
    train, test = sample_users(df, seed=117)
    
    top_content = train[train.tunein.dt.month == m].content_id.value_counts()
    
    # Mnext for test
    df_test_mnext = pd.DataFrame(index=test.account_id.unique())

    # Add seen on Mnext
    df_test_mnext = df_test_mnext.join(
        test[test.tunein.dt.month == m].groupby("account_id")[["content_id"]].agg(set).rename(columns={"content_id": "seen_mnext"}),
        how="left"
    )

    # Fill those who didn't watch anything
    df_test_mnext.loc[df_test_mnext.seen_mnext.isna(), "seen_mnext"] = set()

    def assign_top20_unseen(seen):
        return top_content[~top_content.index.isin(set(seen))].head(20).index.tolist()

    # Make predictions
    df_test_mnext["preds"] = df_test_mnext.seen_mnext.apply(assign_top20_unseen)
    
    # M for test
    df_test_m = pd.DataFrame(index=test.account_id.unique())

    # Add seen on M
    df_test_m = df_test_m.join(
        test[test.tunein.dt.month == m+1].groupby("account_id")[["content_id"]].agg(set).rename(columns={"content_id": "seen_m"}),
        how="left"
    )

    # Fill those who didn't watch anything
    df_test_m.loc[df_test_m.seen_m.isna(), "seen_m"] = set()

    # Required for MAP
    df_test_m["seen_m"] = df_test_m["seen_m"].apply(list)
    
    merged = pd.merge(df_test_m, df_test_mnext, left_index=True, right_index=True, how="outer")

    assert not merged.isna().any().any()
    
    return MAP(
        merged["seen_m"].tolist(),
        merged["preds"].tolist()
    )
    


# %%
map_1 = MAP_1M(1)

# %%
map_2 = MAP_1M(2)

# %%
map_1, map_2, (map_1 + map_2) / 2

# %%
