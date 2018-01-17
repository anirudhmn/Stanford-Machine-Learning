#!/usr/bin/env python
# -*- coding: utf-8 -*-

from basis import WordContainer
from keras.layers import (Input, Dense, Embedding, LSTM, GlobalMaxPool1D,
                          Conv1D, concatenate)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.cross_validation import train_test_split

USE_LSTM = True

# First, open data
with open('sentiment.txt', 'r') as f:
    labels, texts = zip(*[ln.strip().split('|') for ln in f.readlines()])

labels = np.array(list(map(float, labels)))
texts = list(texts)

# Then, load pre-trained vectors
wc = WordContainer.build_from_file('glovevecs.txt', padded=False)

# convert the texts into a tensor of integers
tensor_representation = wc.tensor_transform(texts)
# wc.nearest('') and wc.analogy('','','') Must be all lowercase

words = Input(shape=(tensor_representation.shape[1], ), dtype='int32')

# make a simple model
W = np.random.normal(0, 1, (wc.size + 1, wc.W.shape[1]))

W[1:-1, :] = wc.W

emb = Embedding(
    input_dim=wc.size + 1,
    output_dim=wc.W.shape[1],
    weights=[W],
    mask_zero=USE_LSTM
)

h = emb(words)

if USE_LSTM:

    h = LSTM(30, return_sequences=False)(h)

else:

    h = Conv1D(64, 3, strides=1, activation='relu')(h)
    h = GlobalMaxPool1D()(h)

y = Dense(1, activation='sigmoid')(h)

model = Model(words, y)

model.compile('adam', 'binary_crossentropy', metrics=['acc'])

# train test split
X_train, X_test, y_train, y_test = train_test_split(tensor_representation,
                                                    labels, test_size=0.33)

callbacks = [ModelCheckpoint('tutorial.h5', monitor='val_acc'),
             EarlyStopping(patience=3, monitor='val_acc')]

model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=10,
    callbacks=callbacks,
    validation_data=(X_test, y_test)
)








