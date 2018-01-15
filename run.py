# encoding: utf-8

from __future__ import division, print_function

import cPickle as pickle
import re
import pandas as pd
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU, SpatialDropout1D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import deserialize as layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

import lime
from lime.lime_text import LimeTextExplainer

import ipdb as pdb

class MyTokenizer(BaseEstimator, TransformerMixin):
    '''
    Class for turning text data into sequence of indices
    '''
    def __init__(self, vocab_size=5000, seq_len=33, filters=None):
        if filters is None:
            self.tokenizer = Tokenizer(num_words=None)
        else:
            self.tokenizer = Tokenizer(num_words=None, filters=filters)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.word2idx = None
        self.idx2word = None

    def fit(self, X, y):
        self.tokenizer.fit_on_texts(X)
        self.word2idx = self.tokenizer.word_index
        self.idx2word = {self.word2idx[key]:key for key in self.word2idx}
        return self

    def transform(self, X):
        X = self.tokenizer.texts_to_sequences(X)
        X = np.array([
            [self.vocab_size - 1 if i >= self.vocab_size else i for i in line]
            for line in X])
        X = sequence.pad_sequences(X, maxlen=self.seq_len)
        return X


class MyPipe(object):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def predict_proba(self, X):
        X = self.tokenizer.transform(X)
        prob_1 = self.model.predict_proba(X)
        prob_0 = 1 - prob_1
        return np.concatenate((prob_0, prob_1), axis=1)

class KerasPipeline(object):
    def __init__(self, tokenizer, model, class_names=['Not relevant', 'Relevant']):
        self.tokenizer = tokenizer
        self.model = model
        self.class_names = class_names
        self.for_explanation = None

    def fit(self, X_train, y_train, X_val, y_val, epochs=2):
        self.for_explanation = X_val
        X_train = self.tokenizer.fit_transform(X_train, y_train)
        X_val = self.tokenizer.transform(X_val)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
        batch_size=32)
        return self.model

    def predict_proba(self, X):
        X = self.tokenizer.transform(X)
        prob_1 = self.model.predict_proba(X)
        prob_0 = 1 - prob_1
        return np.concatenate((prob_0, prob_1), axis=1)

    def explain_one_example(self, idx=None, num_features=5, print_out=True):
        if idx is None:
            idx = np.random.choice(self.for_explanation.index)

        explainer = LimeTextExplainer(class_names=self.class_names)
        exp = explainer.explain_instance(self.for_explanation[idx],
                                         self.predict_proba,
                                         num_features=num_features)

        if print_out:
            print ('Tweet {}: {}'.format(idx, self.for_explanation[idx]))
            print (self.predict_proba([self.for_explanation[idx]]))
            print (exp.as_pyplot_figure())
            plt.show()
        return exp

    def explain_model(self, num_examples=10, **kwargs):
        idxs = np.random.choice(self.for_explanation.index, size=num_examples, replace=False)
        contributors = defaultdict(list)
        for i in idxs:
           exp = self.explain_one_example(idx=i, print_out=False, **kwargs)
           for word, weight in exp.as_list():
               contributors[word.lower()].append(weight)
        mean_contrib = []
        for key in contributors:
            mean_contrib.append((key, np.mean(contributors[key])))
        mean_contrib.sort(key=lambda x: np.abs(x[1]), reverse=True)
        return mean_contrib






def baseline(X, y, max_features=5000, f=CountVectorizer):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    vectorizer = f(stop_words='english', decode_error='ignore', max_features=max_features)
    X_train = vectorizer.fit_transform(X_train)

    nb = RandomForestClassifier(n_estimators=500)
    nb.fit(X_train, y_train)

    X_test = vectorizer.transform(X_test)
    print ('Naive Bayes model accuracy: {}'.format(nb.score(X_test, y_test)))
    return vectorizer, nb

def random_f(X, y, max_features=5000, f=CountVectorizer):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    vectorizer = f(stop_words='english', decode_error='ignore', max_features=max_features)
    X_train = vectorizer.fit_transform(X_train)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    X_test = vectorizer.transform(X_test)
    print ('Naive Bayes model accuracy: {}'.format(rf.score(X_test, y_test)))
    return vectorizer, rf


def baseline_grid_search(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe = Pipeline([('vectorizer', CountVectorizer(stop_words='english', decode_error='ignore')),
                  ('nb', MultinomialNB())])

    param_grid = dict(vectorizer__max_features=[5000, 10000, None])

    grid_search = GridSearchCV(pipe,
                                param_grid=param_grid,
                                scoring='accuracy',
                                cv=5,
                                verbose=2,
                                n_jobs=1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def load_w2v(filename):
    '''
    Function to load w2v from filename
    '''
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    return model

def  preprocess(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    #Limit vocab
    vocab_size = 5000
    seq_len=33
    X_train = np.array([[vocab_size - 1 if i >= vocab_size else i for i in line] for line in X_train])
    X_test = np.array([[vocab_size - 1 if i >= vocab_size else i for i in line] for line in X_test])
    X_train = sequence.pad_sequences(X_train, maxlen=seq_len)
    X_test = sequence.pad_sequences(X_test, maxlen=seq_len)
    return tokenizer, X_train, X_test

def cnn(X_train, y_train, X_val, y_val, vocab_size=5000, seq_len=33):
    tokenizer = MyTokenizer()
    X_train = tokenizer.fit_transform(X_train, y_train)
    X_val = tokenizer.transform(X_val)

    model = Sequential([
        Embedding(vocab_size, 32, input_length=seq_len),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.7),
        Dense(1, activation='sigmoid')])

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=4,
    batch_size=32)

    return tokenizer, model



def explain_one_example(tokenizer, model, X_test, idx):
    my_pipe = MyPipe(tokenizer, model)

    explainer = LimeTextExplainer(class_names=['Not relevant', 'Relevant'])

    print ('Tweet: {}'.format(X_test[idx]))
    print (my_pipe.predict_proba(X_test[idx]))

    exp = explainer.explain_instance(X[idx], my_pipe.predict_proba, num_features=5)

    print (exp.as_list())


def train_1_layer_nn():
    pass
    #TODO: function for one layer nn

def create_emb(w2v, tokenizer, vocab_size):
    n_fact = w2v.vector_size
    emb = np.zeros((vocab_size, n_fact))

    for i in range(1,len(emb)):
        word = tokenizer.idx2word[i]
        if word in w2v.vocab:
            emb[i] = w2v[word]
    return emb

def my_pipe_pedict_proba(X, tokenizer, model):
    X = tokenizer.transform(X)
    prob_1 = model.predict_proba(X)[0][0]
    prob_0 = 1 - prob_1
    return np.array([[prob_0, prob_1]])

if __name__ == "__main__":
    df = pd.read_csv("socialmedia-disaster-tweets_clean.csv")
    df = df[df.choose_one != "Can't Decide"]

    X = df.text
    X = X.str.replace('#', '# ')
    X = X.str.replace('@', '@ ')
    X = X.str.replace(r"http\S+", 'http')
    y = pd.get_dummies(df.choose_one, drop_first=True)
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    vocab_size=5000; seq_len=33

    tokenizer = MyTokenizer(vocab_size, seq_len, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n')

    #emb = np.loadtxt('emb.txt')

    model = Sequential([
        Embedding(vocab_size, 32, input_length=seq_len),
        SpatialDropout1D(0.2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.7),
        Dense(1, activation='sigmoid')])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model_embed = Sequential([
    #     Embedding(vocab_size, 50, weights=[emb], input_length=seq_len, trainable=False),
    #     Flatten(),
    #     Dense(100, activation='relu'),
    #     Dropout(0.7),
    #     Dense(1, activation='sigmoid')])
    #
    # model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    #
    #
    # conv1 = Sequential([
    #     Embedding(vocab_size, 25, input_length=seq_len),
    #     SpatialDropout1D(0.2),
    #     Conv1D(64, 5, padding='same', activation='relu'),
    #     Dropout(0.3),
    #     MaxPooling1D(),
    #     Flatten(),
    #     Dense(100, activation='relu'),
    #     Dropout(0.7),
    #     Dense(1, activation='sigmoid')])
    #
    # conv1.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    #
    # #Best conv1 val accuracy: 0.8132
    #
    #
    # conv1emb = Sequential([
    #     Embedding(vocab_size, 50, input_length=seq_len, weights=[emb], trainable=False),
    #     SpatialDropout1D(0.2),
    #     Conv1D(128, 5, padding='same', activation='relu'),
    #     Dropout(0.5),
    #     MaxPooling1D(),
    #     Flatten(),
    #     Dense(100, activation='relu'),
    #     Dropout(0.7),
    #     Dense(1, activation='sigmoid')])
    #
    # conv1emb.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    #
    # #Best conv1emb val accuracy: 0.8020
    #
    # conv1emb0_1 = Sequential([
    #     Embedding(vocab_size, 50, input_length=seq_len, weights=[emb], trainable=False),
    #     Conv1D(128, 5, padding='same', activation='relu'),
    #     MaxPooling1D(5),
    #     Conv1D(128, 5, padding='same', activation='relu'),
    #     MaxPooling1D(),
    #     Conv1D(128, 5, padding='same', activation='relu'),
    #     MaxPooling1D(),
    #     Flatten(),
    #     Dense(100, activation='relu'),
    #     Dropout(0.7),
    #     Dense(1, activation='sigmoid')])

    # conv1emb.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # rnn = Sequential([
    # (Embedding(vocab_size, 50, weights=[emb], input_length=seq_len, trainable=False)),
    # (LSTM(100, go_backwards=True)),
    # (Dense(1, activation='sigmoid'))])
    #
    # rnn.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


    pipe = KerasPipeline(tokenizer, model)

    pipe.fit(X_train, y_train, X_test, y_test, epochs=2)


    pipe.explain_one_example(9578)

    filename='/Users/yauhenikoran/Glove/glove.twitter.27B.50d.w2v.txt'
