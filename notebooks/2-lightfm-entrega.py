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

from utils import load_data, MAP

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

from lightfm import LightFM
from lightfm.evaluation import auc_score as lfm_auc_score

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=16)

from tqdm import tqdm

tqdm.pandas()

# %% [markdown]
# # Pre-procesamiento de datos

# %%
AGG = "sum"

# %%
df, metadata = load_data()

all_users = np.arange(df.account_id.min(), df.account_id.max()+1).astype(np.int32)
all_items = np.arange(metadata.content_id.min(), metadata.content_id.max()+1).astype(np.int32)

df.dropna(inplace=True, subset=["content_id", "account_id"])
df["dummy"] = 1

df.account_id = df.account_id.astype(np.int32)
df.content_id = df.content_id.astype(np.int32)

# %%
metadata_sub = metadata[
    ["show_type", "released_year", "country_of_origin", "category",
     "run_time_min", "audience", "made_for_tv", "pay_per_view",
     "pack_premium_1", "pack_premium_2",
     # "create_date", "modify_date","start_vod_date", "end_vod_date"
    ]
]

metadata_sub = pd.get_dummies(
    metadata_sub,
    columns=["show_type", "released_year", "country_of_origin", "category",
     "audience", "made_for_tv", "pay_per_view",
     "pack_premium_1", "pack_premium_2"]
)

# for col in ["create_date", "modify_date", "start_vod_date", "end_vod_date"]:
#     metadata_sub[f"{col}_year"] = metadata_sub[col].dt.year
#     metadata_sub[f"{col}_month"] = metadata_sub[col].dt.month

#metadata_sub.drop(columns=["create_date", "modify_date", "start_vod_date", "end_vod_date"], inplace=True)

metadata_sub = scipy.sparse.csr_matrix(metadata_sub.values)

# %%
df_train = df[df.tuneout.dt.month < 3]
df_test = df[df.tuneout.dt.month == 3]

# %%
trainset_train_m = df_train.pivot_table(
    index="account_id",
    columns="content_id",
    values="dummy",
    aggfunc=AGG,
    fill_value=0,
)

# %%
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
N_COMPONENTS = 500
MAX_SAMPLED = 100
LR = 0.05
EPOCHS = 5

# %%
model = LightFM(no_components=N_COMPONENTS, loss="warp", learning_schedule="adagrad", learning_rate=LR, max_sampled=MAX_SAMPLED)

# %%
model.fit(
    scipy.sparse.csr_matrix(trainset_train_m.values),
    epochs=EPOCHS,
    verbose=True,
    num_threads=1,
    #item_features=metadata_sub,
    # item_alpha=1e-6,
)

# %% [markdown]
# # Test

# %%
testset_m = df_test.pivot_table(
    index="account_id",
    columns="content_id",
    values="dummy",
    aggfunc=AGG,
    fill_value=0,
)

# %%
zero_fill = pd.DataFrame(
    0,
    index=list(set(all_users) - set(testset_m.index)),
    columns=testset_m.columns,
)
testset_m = pd.concat([testset_m, zero_fill])
testset_m[list(set(all_items) - set(testset_m.columns))] = 0
testset_m = testset_m.sort_index()
testset_m = testset_m[sorted(testset_m.columns)]

testset_m.index = testset_m.index.astype(np.int32)

# %%
lfm_auc_score(
    model,
    scipy.sparse.csr_matrix(testset_m),
    num_threads=1,
    # item_features=metadata_sub
).mean()


# %%
def predict_for_user(user_id):
    items_seen_train = df_train[df_train.account_id == user_id].content_id.unique()
    items_pred = sorted([i for i in all_items if i not in items_seen_train])

    preds = (
        pd.DataFrame(
            {"item": items_pred, "pred": model.predict(
                int(user_id),
                items_pred,
                num_threads=1,
                #item_features=metadata_sub
            )}
        )
        .sort_values(by="pred", ascending=False)
        .head(20)
        .item.tolist()
    )

    return preds


z = pd.DataFrame({"user_id": all_users})
z["recs"] = z.user_id.parallel_apply(predict_for_user)
z = z.sort_values("user_id")

ground_truth = pd.DataFrame(df_test.groupby("account_id").content_id.agg(list).sort_index()).rename(columns={"content_id": "seen"})

mix = pd.merge(z, ground_truth, left_index=True, right_index=True, how="left")
mix["seen"] = mix.seen.apply(lambda x: x if isinstance(x, list) else [])

MAP(mix.seen, mix.recs)

# %% [markdown]
# # Predicciones de items no vistos 

# %%
df, metadata = load_data()

all_users = np.arange(df.account_id.min(), df.account_id.max()+1).astype(np.int32)
all_items = np.arange(metadata.content_id.min(), metadata.content_id.max()+1).astype(np.int32)

df.dropna(inplace=True, subset=["content_id", "account_id"])
df["dummy"] = 1

df.account_id = df.account_id.astype(np.int32)
df.content_id = df.content_id.astype(np.int32)

# %%
m = df.pivot_table(
    index="account_id",
    columns="content_id",
    values="dummy",
    aggfunc=AGG,
    fill_value=0,
)

# %%
m.shape

# %%
zero_fill = pd.DataFrame(
    0,
    index=list(set(all_users) - set(m.index.values)),
    columns=m.columns,
)
m = pd.concat([m, zero_fill])
m[list(set(all_items) - set(m.columns))] = 0

m = m.sort_index()
m = m[sorted(m.columns)]

m.index = m.index.astype(np.int32)

# %%
m.shape

# %%
model = LightFM(no_components=N_COMPONENTS, loss="warp", learning_schedule="adagrad", learning_rate=LR, max_sampled=MAX_SAMPLED)

# %%
model.fit(
    scipy.sparse.csr_matrix(m.values),
    epochs=EPOCHS,
    verbose=True,
    num_threads=1,
    #item_features=metadata_sub,
    # item_alpha=1e-6,
)


# %%
def predict_for_user(user_id):
    items_seen = df[df.account_id == user_id].content_id.unique()
    items_pred = sorted([i for i in all_items if i not in items_seen])

    preds = (
        pd.DataFrame(
            {"item": items_pred, "pred": model.predict(
                int(user_id),
                items_pred
            )}
        )
        .sort_values(by="pred", ascending=False)
        .head(20)
        .item.tolist()
    )

    return preds


z = pd.DataFrame({"user_id": all_users})
z["recs"] = z.user_id.parallel_apply(predict_for_user)
z = z.sort_values("user_id")

# %%
z["recs"] = z.recs.apply(json.dumps)

# %%
z.to_csv("../entregas/2-lightfm-2.csv", index=False, escapechar='"', header=False)

# %%
assert len(z) == df.account_id.max() + 1

# %%
z

# %%
