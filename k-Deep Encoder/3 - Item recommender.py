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
from datetime import datetime
from dateutil import parser

# +
from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
get_available_devices()
# -

watch_data = pd.read_csv('../data/train.csv')

metadata = pd.read_csv('../data/metadata.csv', sep=';')
metadata = metadata[metadata['content_id'].notnull()]
asset_to_content = metadata.set_index('asset_id')['content_id'].to_dict()

train_df, test_df = sample_users(watch_data)
train_df = train_df[train_df['asset_id'].notnull()]
train_df = train_df[train_df['resume']==0]
train_df['month'] = train_df['tunein'].map(lambda x: int(x.split('-')[1][-1]))
train_df['asset_id'] = train_df['asset_id'].map(lambda x: [int(asset_to_content[x])] if x in asset_to_content else None)
train_df = train_df[train_df['asset_id'].notnull()]

train_df.sample(5)

# +
import pickle

vectors = np.load('encoded_vectors.npy')
with open('content_ids', 'rb') as file:
    content_ids = pickle.load(file)
# -

ids_map = {cid:i for i,cid in enumerate(content_ids)}


def dataframe_contents_per_month(df, months):
    months_df = df[df['month'].isin(months)]
    months_df = months_df.sort_values('tunein')
    return months_df.groupby('customer_id').agg({'asset_id':'sum'})['asset_id'].map(lambda x: list(dict.fromkeys(x))).to_dict()


months_12 = train_df[train_df['month'].isin({1,2})]
months_12 = months_12.sort_values('tunein')
contents_months12 = dataframe_contents_per_month(train_df, {1,2})
months_12['asset_id'] = months_12['asset_id'].map(lambda x: x[0])

reproduction_popularity = (months_12.groupby('asset_id').agg({'customer_id': 'count'})['customer_id']/len(months_12))
user_popularity = (months_12.groupby('asset_id').agg({'customer_id': 'nunique'})['customer_id']/months_12['customer_id'].nunique())

release_delta_3 = (pd.to_datetime(metadata.groupby('content_id').agg({'start_vod_date':'max'})['start_vod_date'])-parser.parse('2021-03-01T00:00:00.0Z')).map(lambda x: x.days)

normalization_date_param = release_delta_3.mean()

release_delta_3 = release_delta_3/normalization_date_param
release_delta_3_dict = release_delta_3.to_dict()

contents_months3 = dataframe_contents_per_month(train_df, {3})


# ### features extra al vector:
# * tiempo que esta en la plataforma
# * popularidad por reproduccion
# * popularidad por usuarios

def data_generator(contents_previous_months, contents_actual_months, 
                   reproduction_popularity_dict,
                   user_popularity_dict,
                   release_delta_dict,
                   random_per_cust=4, batch_size=1024):
    contents_actual_months = {k:v for k,v in contents_actual_months.items() if v}
    content_ids = list(release_delta_dict.keys())
    customer_ids = list(set(contents_actual_months.keys()).intersection(set(contents_previous_months.keys())))
    for k in content_ids:
        if k not in reproduction_popularity_dict:
            reproduction_popularity_dict[k] = 0
        if k not in user_popularity_dict:
            user_popularity_dict[k] = 0
    while True:
        past_sequences = []
        future_sequences = []
        label_map = []
        positive_examples = batch_size//random_per_cust
        for _ in range(positive_examples):
            customer_choice = random.choice(customer_ids)
            index_to_use = random.randint(1,len(contents_actual_months[customer_choice]))
            output_prediction_list = contents_actual_months[customer_choice][:index_to_use]
            input_prediction_list = (contents_previous_months[customer_choice] if customer_choice in contents_previous_months else [])
            output_prediction_list = [elem for elem in output_prediction_list if elem not in input_prediction_list]
            past_sequences.append(input_prediction_list)
            future_sequences.append(output_prediction_list)
            label_map.append(1)
            fake_sequence=output_prediction_list
            for __ in range(random_per_cust-1):
                random_content = random.choice(content_ids)
                past_sequences.append(input_prediction_list)
                fake_sequence = fake_sequence+[random_content]
                random.shuffle(fake_sequence)
                future_sequences.append(fake_sequence)
                label_map.append(0)
        past_sequences = [[vectors[ids_map[cid]].tolist()+\
                           [reproduction_popularity_dict[cid]]+\
                           [user_popularity_dict[cid]]+\
                           [release_delta_dict[cid]] for cid in watched] for watched in past_sequences]
        future_sequences = [[vectors[ids_map[cid]].tolist()+\
                           [reproduction_popularity_dict[cid]]+\
                           [user_popularity_dict[cid]]+\
                           [release_delta_dict[cid]] for cid in watched] for watched in future_sequences]
        past_sequences = pad_sequences(past_sequences, maxlen=20, dtype='float32', padding='post',truncating='post', value=0.0)
        future_sequences = pad_sequences(future_sequences, maxlen=20, dtype='float32', padding='post',truncating='post', value=0.0)
        yield [past_sequences, future_sequences], np.asarray(label_map)


# # Train

# +
initial_shape = vectors.shape[1]+3

# DNN encoder

dnn_encoder_input = keras.layers.Input((3*initial_shape,))
dout = keras.layers.Dropout(0.05)(dnn_encoder_input)
dense = keras.layers.Dense(50, activation='tanh')(dout)
dout = keras.layers.Dropout(0.05)(dense)
dense = keras.layers.Dense(50, activation='tanh')(dout)
dout = keras.layers.Dropout(0.05)(dense)
dense = keras.layers.Dense(50, activation='tanh')(dout)
dout = keras.layers.Dropout(0.05)(dense)
dense = keras.layers.Dense(50, activation='tanh')(dout)

dnn_encoder_model = keras.Model(dnn_encoder_input, dense)

# Main model

encoder_input = keras.layers.Input((20, initial_shape))
decoder_output = keras.layers.Input((20, initial_shape))

initial_conv = keras.layers.Conv1D(initial_shape, 1)

input_conved = initial_conv(encoder_input)
output_conved = initial_conv(decoder_output)

attention_seq = tf.keras.layers.Attention()([keras.layers.Masking(0.0)(input_conved), keras.layers.Masking(0.0)(output_conved)])

attention_seq_avg = keras.layers.GlobalAveragePooling1D()(attention_seq)
attention_seq_max = keras.layers.GlobalMaxPooling1D()(attention_seq)
input_max = keras.layers.GlobalMaxPooling1D()(input_conved)
output_max = keras.layers.GlobalMaxPooling1D()(output_conved)

input_seq_encoded = keras.layers.Concatenate()([attention_seq_avg, attention_seq_max, input_max]) # 33 + 33 + 33
output_seq_encoded = keras.layers.Concatenate()([attention_seq_avg, attention_seq_max, output_max])

# Predictor model

encoding1 = dnn_encoder_model(input_seq_encoded)
encoding2 = dnn_encoder_model(output_seq_encoded)

subtracted = keras.layers.Subtract()([encoding1, encoding2])
multiply = keras.layers.Multiply()([encoding1, encoding2])
concatenate = keras.layers.Concatenate()([subtracted, multiply])
result = keras.layers.Dense(1, activation='sigmoid')(concatenate)

# Trainable model

model = keras.Model([encoder_input, decoder_output], result)
# -

model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='binary_crossentropy', metrics=['binary_accuracy'])

generator = data_generator(contents_months12, contents_months3, 
                           reproduction_popularity,
                           user_popularity,
                           release_delta_3, batch_size=2048)

# +
test_df = test_df[test_df['asset_id'].notnull()]
test_df = test_df[test_df['resume']==0]
test_df['month'] = test_df['tunein'].map(lambda x: int(x.split('-')[1][-1]))
test_df['asset_id'] = test_df['asset_id'].map(lambda x: [int(asset_to_content[x])] if x in asset_to_content else None)
test_df = test_df[test_df['asset_id'].notnull()]

test_contents_months12 = dataframe_contents_per_month(test_df, {1,2})

test_contents_months3 = dataframe_contents_per_month(test_df, {3})
# -

valid_generator = data_generator(test_contents_months12, test_contents_months3, 
                                 reproduction_popularity,
                                 user_popularity,
                                 release_delta_3, batch_size=2048)

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)

history = model.fit(generator, steps_per_epoch=50, epochs=500, batch_size=2048, callbacks=[callback], 
                    validation_data=valid_generator, validation_steps=20)

plt.figure(dpi=150)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'])
plt.ylim((0,0.5))
plt.show()

model.save('predictor_model')

# # Load model

model = keras.models.load_model('predictor_model')

content_ids = set(metadata['content_id'])

from tqdm import tqdm


def get_sequences_predict(input_prediction_list, actual_sequence, not_seen):
    sequences_to_use = []
    for content in not_seen:
        sequences_to_use.append(actual_sequence+[content])
    sequences_to_use_vector = [[vectors[ids_map[cid]].tolist()+\
                           [(reproduction_popularity[cid] if cid in reproduction_popularity else 0)]+\
                           [(user_popularity[cid] if cid in reproduction_popularity else 0)]+\
                           [release_delta_3_dict[cid]] for cid in candidate] for candidate in sequences_to_use]
    if input_prediction_list:
        past_sequences = [input_prediction_list]*len(sequences_to_use)
        past_sequences = [[vectors[ids_map[cid]].tolist()+\
                           [(reproduction_popularity[cid] if cid in reproduction_popularity else 0)]+\
                           [(user_popularity[cid] if cid in reproduction_popularity else 0)]+\
                           [release_delta_3_dict[cid]] for cid in candidate] for candidate in past_sequences]
        past_sequences = pad_sequences(past_sequences, maxlen=20, dtype='float32', padding='post',truncating='post', value=0.0)
    else:
        past_sequences = np.zeros((len(sequences_to_use), 20, 33))
    sequences_to_use_vector = pad_sequences(sequences_to_use_vector, maxlen=20, dtype='float32', padding='post',truncating='post', value=0.0)
    return past_sequences, sequences_to_use_vector, sequences_to_use


vectors.shape

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=50, algorithm='brute').fit(vectors)
_, neigh_indices = nbrs.kneighbors(vectors)

neigh_indices = neigh_indices.tolist()

cid_to_index = ids_map
index_to_cid = {v:k for k,v in ids_map.items()}

most_popular_cids = user_popularity.sort_values(ascending=False).index.tolist()[:200]

max_date_content = metadata.groupby('content_id').agg({'start_vod_date':'max'})
new_contents = max_date_content[pd.to_datetime(max_date_content['start_vod_date'])>parser.parse('2021-03-01T00:00:00.0Z')].index.tolist()

predictions = {}
keys_for_prediction_sample = list(set(test_contents_months12.keys()))
random.shuffle(keys_for_prediction_sample)
for k in tqdm(keys_for_prediction_sample[:200]):
#for k1,k2,k3,k4 in tqdm(zip(*[iter(keys_for_prediction_sample)]*4), total=(keys_for_prediction_sample//4+keys_for_prediction_sample%4)):
    input_prediction_list = test_contents_months12[k]
    not_seen = set(most_popular_cids+new_contents)
    for cid in input_prediction_list:
        not_seen.update([index_to_cid[index] for index in neigh_indices[cid_to_index[cid]]])
    not_seen = not_seen-set(input_prediction_list)
    actual_sequence = []
    for _ in range(20):
        past_sequences, sequences_to_use_vector, sequences_to_use = get_sequences_predict(input_prediction_list, actual_sequence, not_seen)
        preds = model.predict([past_sequences, sequences_to_use_vector])
        actual_sequence = sequences_to_use[preds.flatten().argmax()]
        not_seen = not_seen - set(actual_sequence)
    predictions[k] = actual_sequence

# +
true_sequences = []
pred_sequences = []

for k in predictions.keys():
    if k in test_contents_months3:
        true_sequences.append(test_contents_months3[k])
    else:
        true_sequences.append([])
    pred_sequences.append(predictions[k])
    seen = test_contents_months12[k]
    popular_cids_not_seen = [cid for cid in most_popular_cids if cid not in seen]
    pred_sequences[-1]+=popular_cids_not_seen[:0]
MAP(true_sequences, pred_sequences)
# -

