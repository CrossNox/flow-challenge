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

# %% [markdown]
# # Importamos bibliotecas
# %%
import numpy as np

import pandas as pd

from utils import load_data

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import json

import lightfm
import numpy as np
import scipy
from progressbar import progressbar
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas()

# %% [markdown]
# # Pre-procesamiento de datos

# %%
df, metadata = load_data()

df.dropna(inplace=True, subset=["content_id", "account_id"])

df.account_id = df.account_id.astype(np.int32)
df.content_id = df.content_id.astype(np.int32)

# %%
trainset_train_m = df.pivot_table(
    index="account_id",
    columns="content_id",
    values="asset_id",
    aggfunc="count",
    fill_value=0,
)

# %%
all_items = np.arange(metadata.content_id.min(), metadata.content_id.max() + 1)
all_users = np.arange(df.account_id.min(), df.account_id.max() + 1)

zero_fill = pd.DataFrame(
    0,
    index=list(set(all_users) - set(trainset_train_m.index)),
    columns=trainset_train_m.columns,
)
trainset_train_m = pd.concat([trainset_train_m, zero_fill])
trainset_train_m[list(set(all_items) - set(trainset_train_m.columns))] = 0
trainset_train_m = trainset_train_m.sort_index()
trainset_train_m = trainset_train_m[sorted(trainset_train_m.columns)]

trainset_train_m.index = trainset_train_m.index.astype(np.int32)

# %% [markdown]
# # Entrenamos modelo

# %%
from lightfm import LightFM

model = LightFM(no_components=100, loss="warp", learning_schedule="adagrad")

# %%
model.fit(scipy.sparse.csr_matrix(trainset_train_m.values), epochs=500, verbose=True)


# %% [markdown]
# # Predicciones de items no vistos 

# %%
def predict_for_user(user_id):
    all_items = metadata.content_id.dropna().unique()
    items_seen = df[df.account_id == user_id].content_id.unique()
    items_pred = np.array(list(set(all_items) - set(items_seen))).astype(np.int32)

    preds = (
        pd.DataFrame(
            {"item": items_pred, "pred": model.predict(int(user_id), items_pred)}
        )
        .sort_values(by="pred", ascending=False)
        .head(20)
        .item.tolist()
    )

    return preds


z = pd.DataFrame({"user_id": df.account_id.drop_duplicates().sort_values().unique()})
z["recs"] = z.user_id.progress_apply(predict_for_user)
z = z.sort_values("user_id")

# %%
z["recs"] = z.recs.apply(json.dumps)

# %%
z.to_csv("../entregas/2-lightfm.csv", index=False, escapechar='"', header=False)

# %%
