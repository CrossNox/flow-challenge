import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.keras.constraints import non_neg
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from utils import sample_users, MAP
import matplotlib.pyplot as plt

# +
from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
get_available_devices()
# -

watch_data = pd.read_csv('../data/train.csv')

metadata = pd.read_csv('../data/metadata.csv', sep=';')

train_df, _ = sample_users(watch_data)

metadata.content_id.value_counts()

metadata.columns

metadata.cast_first_name

# # Mappings para el encoder

# * show_type: one hot
# * released_year: mean
# * country_of_origin: one hot para los mas comunes
# * category: split '/' bow -> PCA
# * keywords: split ',' bow -> PCA
# * cast_first_name: bow split ',' -> PCA
# * run_time_min: mean and median
# * audience: fillna, lowercase y one hot
# * 'made_for_tv', 'pay_per_view', 'pack_premium_1', 'pack_premium_2': mean

# +
binary_cats = ['made_for_tv', 'pay_per_view', 'pack_premium_1', 'pack_premium_2']

metadata_preprocess = metadata.copy()
metadata_preprocess['show_type'] = metadata_preprocess['show_type'].fillna('')
metadata_preprocess['country_of_origin'] = metadata_preprocess['country_of_origin'].fillna("")
metadata_preprocess['category'] = metadata_preprocess['category'].fillna("").map(lambda x: x.split('/')  if x else [])
metadata_preprocess['keywords'] = metadata_preprocess['keywords'].fillna("").map(lambda x: x.split(',')  if x else [])
metadata_preprocess['cast_first_name'] = metadata_preprocess['cast_first_name'].fillna("").map(lambda x: x.split(',') if x else [])
metadata_preprocess['audience'] = metadata_preprocess['audience'].fillna("").str.lower()
metadata_preprocess[binary_cats] = metadata_preprocess[binary_cats].fillna('N').applymap(lambda x:  x=='Y')
metadata_preprocess.sample(5)
# -

content_metadata = metadata_preprocess.groupby('content_id').agg({'show_type': 'max',
                                                                  'released_year': 'mean',
                                                                  'country_of_origin': 'max',
                                                                  'category': sum,
                                                                  'keywords': sum,
                                                                  'cast_first_name': sum,
                                                                  'audience': 'max',
                                                                  'run_time_min': ['mean', 'median'],
                                                                  'made_for_tv': 'mean', 
                                                                  'pay_per_view': 'mean', 
                                                                  'pack_premium_1': 'mean', 
                                                                  'pack_premium_2': 'mean'
                                                                 })
content_metadata.columns = ['%s%s' % (a, '-%s' % b if b else '') for a, b in content_metadata.columns]
content_metadata = content_metadata.reset_index()
content_metadata['category-sum'] = content_metadata['category-sum'].map(lambda x: list(set(x)))
content_metadata['keywords-sum'] = content_metadata['keywords-sum'].map(lambda x: list(set(x)))

content_metadata.sample(5)

# # Encoding show type

from sklearn.preprocessing import OneHotEncoder

content_metadata['show_type-max'].value_counts()

content_metadata['show_type-max'].value_counts().index[:4].tolist()

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(content_metadata['show_type-max'].value_counts().index[:4].values.reshape(-1, 1))
show_types = encoder.transform(content_metadata['show_type-max'].values.reshape(-1, 1))
show_types.shape

# # Encoding country

series = content_metadata['country_of_origin-max'].value_counts()
series = series[series>50]
series

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(series.index.values.reshape(-1, 1))
country = encoder.transform(content_metadata['country_of_origin-max'].values.reshape(-1, 1))
country.shape

# # Bow category

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

vectorizer = CountVectorizer(preprocessor=lambda x: [c.lower() for c in x],tokenizer=lambda x: x, lowercase=False)
category_bow = vectorizer.fit_transform(content_metadata['category-sum'])
category_bow.shape

svd = TruncatedSVD(n_components=47)
category_pca = svd.fit_transform(category_bow)

plt.plot(svd.explained_variance_)

svd = TruncatedSVD(n_components=20)
category_pca = svd.fit_transform(category_bow)
category_pca.shape

# # Bow keywords

vectorizer = CountVectorizer(preprocessor=lambda x: [c.lower() for c in x],tokenizer=lambda x: x, lowercase=False)
keywords_bow = vectorizer.fit_transform(content_metadata['keywords-sum'])
keywords_bow.shape

svd = TruncatedSVD(n_components=320)
keywords_pca = svd.fit_transform(keywords_bow)

plt.plot(svd.explained_variance_)

svd = TruncatedSVD(n_components=40)
keywords_pca = svd.fit_transform(keywords_bow)
keywords_pca.shape

svd.explained_variance_ratio_.sum()

# # Bow cast

vectorizer = CountVectorizer(preprocessor=lambda x: [c.lower() for c in x],tokenizer=lambda x: x, lowercase=False)
cast_bow = vectorizer.fit_transform(content_metadata['cast_first_name-sum'])
cast_bow.shape

svd = TruncatedSVD(n_components=200)
cast_pca = svd.fit_transform(cast_bow)

plt.plot(svd.explained_variance_)

svd = TruncatedSVD(n_components=25)
cast_pca = svd.fit_transform(cast_bow)
cast_pca.shape

svd.explained_variance_ratio_.sum()

# # Encoding audience

content_metadata['audience-max'].value_counts()

content_metadata['audience-max'].value_counts().index[:-1].tolist()

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(content_metadata['audience-max'].value_counts().index[:-1].values.reshape(-1, 1))
audience_types = encoder.transform(content_metadata['audience-max'].values.reshape(-1, 1))
audience_types.shape

# # Numerical feats

content_metadata.columns

numerical_feats = content_metadata[['released_year-mean', 'run_time_min-mean',
                                   'run_time_min-median', 'made_for_tv-mean', 'pay_per_view-mean',
                                   'pack_premium_1-mean', 'pack_premium_2-mean']].values
numerical_feats.shape

# # Concat all

from sklearn.preprocessing import StandardScaler
import numpy as np

numerical = np.concatenate([category_pca, keywords_pca, cast_pca, numerical_feats], axis=1)
numerical.shape

numerical = StandardScaler().fit_transform(numerical)
numerical.shape

vectors = np.concatenate([show_types.todense(), country.todense(), audience_types.todense(), numerical], axis=1)
vectors.shape

vectors

np.save('vectors', vectors)

import pickle
with open('content_ids', 'wb') as file:
    pickle.dump(content_metadata['content_id'].map(lambda x: int(x)).tolist(), file)


