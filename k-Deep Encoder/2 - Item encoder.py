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
import numpy as np
import random

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

train_df.sample(5)

metadata = metadata[metadata['content_id'].notnull()]
asset_to_content = metadata.set_index('asset_id')['content_id'].to_dict()

train_df = train_df[['customer_id', 'asset_id']]
train_df = train_df[train_df['asset_id'].notnull()]
train_df['asset_id'] = train_df['asset_id'].map(lambda x: [int(asset_to_content[x])] if x in asset_to_content else [])

customer_choices = train_df.groupby('customer_id').agg({'asset_id':'sum'})
customer_choices['asset_id'] = customer_choices['asset_id'].map(lambda x: list(set(x)))
customer_choices.sample(5)

customer_choices = customer_choices.values.tolist()

customer_choices[:5]

customer_choices = [choices[0] for choices in customer_choices if len(choices[0])>1]
len(customer_choices)

customer_choices[:5]

# +
import pickle

vectors = np.load('vectors.npy')
with open('content_ids', 'rb') as file:
    content_ids = pickle.load(file)
# -

ids_map = {cid:i for i,cid in enumerate(content_ids)}


def data_generator(n_neg=4, batch_size=1024):
    while True:
        vectors1 = []
        vectors2 = []
        content_ids = list(ids_map.keys())
        positive_examples = batch_size//n_neg
        negative_examples = batch_size - positive_examples
        for _ in range(positive_examples):
            random_list = random.choice(customer_choices).copy()
            random.shuffle(random_list)
            vectors1.append(random_list[0])
            vectors2.append(random_list[1])
            for __ in range(n_neg-1):
                vectors1.append(random_list[0])
                vectors2.append(random.choice(content_ids))
        labels = ([1]+[0]*(n_neg-1))*positive_examples
        vectors1 = [vectors[ids_map[cid]] for cid in vectors1]
        vectors2 = [vectors[ids_map[cid]] for cid in vectors2]
        yield [np.asarray(vectors1), np.asarray(vectors2)], np.asarray(labels)


# +
input1 = keras.layers.Input((vectors.shape[1],))
input2 = keras.layers.Input((vectors.shape[1],))

# encoder

encoder_input = keras.layers.Input((vectors.shape[1],))
dout = keras.layers.Dropout(0.05)(encoder_input)
dense = keras.layers.Dense(30, activation='tanh')(dout)
dout = keras.layers.Dropout(0.05)(dense)
dense = keras.layers.Dense(30, activation='tanh')(dout)
dout = keras.layers.Dropout(0.05)(dense)
dense = keras.layers.Dense(30, activation='tanh')(dout)
dout = keras.layers.Dropout(0.05)(dense)
dense = keras.layers.Dense(30, activation='tanh')(dout)

encoder_model = keras.Model(encoder_input, dense)

# Predictor model

encoding1 = encoder_model(input1)
encoding2 = encoder_model(input2)

subtracted = keras.layers.Subtract()([encoding1, encoding2])
multiply = keras.layers.Multiply()([encoding1, encoding2])
concatenate = keras.layers.Concatenate()([subtracted, multiply])
result = keras.layers.Dense(1, activation='sigmoid')(concatenate)

# Trainable model

model = keras.Model([input1, input2], result)
# -

model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='binary_crossentropy')

generator = data_generator(n_neg=4, batch_size=4096)

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=300)

history = model.fit(generator, steps_per_epoch=50, epochs=10000, batch_size=4096, callbacks=[callback])

generator = data_generator(n_neg=4)

preds = model.predict(generator, steps=10).flatten().tolist()

positive_preds = [preds[i] for i in range(len(preds)) if i%4==0]
negative_preds = [preds[i] for i in range(len(preds)) if i%4!=0]

np.median(positive_preds)

np.median(negative_preds)

plt.hist(positive_preds)

plt.hist(negative_preds)

encodeds = encoder_model.predict(vectors)

encodeds.shape

from sklearn.manifold import TSNE

tsne_encoder = TSNE(n_components=2)
tsne_encodeds = tsne_encoder.fit_transform(encodeds)

tsne_encodeds.shape

import seaborn as sns

plt.figure(dpi=150)
plt.scatter(tsne_encodeds[:,0], tsne_encodeds[:,1], s=1)

np.save('encoded_vectors.npy', encodeds)

encoder_model.save('encoder')


