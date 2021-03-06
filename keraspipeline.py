# encoding: utf-8
from __future__ import division, print_function

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam

from lime.lime_text import LimeTextExplainer

class MyTokenizer(BaseEstimator, TransformerMixin):
    '''
    Class for turning text data into sequence of indices.

    Wrapper of keras.preprocessing.text.Tokenizer. Additionally limits vocabulary
    and pre-pads sequences with zeros to the unified length.

    # Arguments:
        vocab_size: int, size of the vocabulary (default: 5000).
        seq_len: int, unifies length (default: 33).
        kwargs: arguments to keras.preprocessing.text.Tokenizer.
    '''
    def __init__(self, vocab_size=5000, seq_len=33, **kwargs):
        self.tokenizer = Tokenizer(**kwargs)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.word2idx = None
        self.idx2word = None

    def fit(self, X, y):
        '''Updates internal vocabulary based on a list of texts.

        # Arguments:
            X: sequence of strings.
            y: sequnce of ints or booleans.

        # Returns:
            self
        '''
        self.tokenizer.fit_on_texts(X)
        self.word2idx = self.tokenizer.word_index
        self.idx2word = {self.word2idx[key]:key for key in self.word2idx}
        return self

    def transform(self, X):
        '''Converts sequence of strings to a Numpy ndarray.

        #Arguments:
            X: sequence of strings.

        # Returns:
            X: Numpy ndarray.
        '''
        X = self.tokenizer.texts_to_sequences(X)
        X = np.array([
            [self.vocab_size - 1 if i >= self.vocab_size else i for i in line]
            for line in X])
        X = sequence.pad_sequences(X, maxlen=self.seq_len)
        return X


class KerasPipeline(object):
    '''
    This class allows to create, train and assess keras models. KerasPipeline
    combines MyTokenizer and Keras models into a single class, making it easy to
    experiment with model architectures.

    User chooses parameters of embeddings and model type, internal model
    parameters are fixed.

    # Arguments:
        model: string, model type (choice of: 'nn', 'cnn1d', 'cnn1d_emb','lstm').
        class_names: list of strings, names of classes.
        vocab_size: int, size of the vocabulary (default: 5000).
        embed_size: int, size of the word embeddings (default: 32).
        seq_len: int, unified length of data representation (default: 33).
        emb_path: string, path to the pre-trained embeddings vectores. If
            specified, rewrites vocab_size and embed_size to the parameters of
            pre-trained vectors.

    '''
    def __init__(self, model, class_names=['Not relevant', 'Relevant'],
           vocab_size=5000, embed_size=32, seq_len=33, emb_path=None):

        assert model in ['nn', 'cnn1d', 'cnn1d_emb', 'lstm'], \
            "Invalid model: '{}'. Available models to train: ['nn', 'cnn1d', 'cnn1d_emb', 'lstm']".format(model)

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.class_names = class_names
        self.for_explanation = None #Save validation data for explanations
        self.emb_path = emb_path
        self.emb = None #Embeddings matrix

        #Load embeddings from file if model needs them
        if model in ['cnn1d_emb', 'lstm']:
            assert self.emb_path is not None, 'No path to embeddings file'
            self.emb = np.loadtxt(self.emb_path)
            self.vocab_size, self.embed_size = self.emb.shape

        self.tokenizer = MyTokenizer(self.vocab_size,
                                     self.seq_len,
                                     filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n')

        self.model = self.load_model(model)

    def load_model(self, model):
        '''Loads Keras model and prints its summary.

        # Arguments:
            model: string, model type.
        # Returns:
            model: compiled Keras model
        '''

        #Raname variables for briefness
        V, E, S = self.vocab_size, self.embed_size, self.seq_len

        if model == 'nn':

            model = Sequential([
                Embedding(V, E, input_length=S),
                SpatialDropout1D(0.2),
                Flatten(),
                Dense(100, activation='relu'),
                Dropout(0.7),
                Dense(1, activation='sigmoid')])

        elif model == 'cnn1d':

            model = Sequential([
                Embedding(V, E, input_length=S),
                SpatialDropout1D(0.2),
                Conv1D(64, 5, padding='same', activation='relu'),
                Dropout(0.3),
                MaxPooling1D(),
                Flatten(),
                Dense(100, activation='relu'),
                Dropout(0.7),
                Dense(1, activation='sigmoid')])

        elif model == 'cnn1d_emb':

            model = Sequential([
                Embedding(V, E, input_length=S , weights=[self.emb], trainable=False),
                SpatialDropout1D(0.2),
                Conv1D(128, 5, padding='same', activation='relu'),
                Dropout(0.5),
                MaxPooling1D(),
                Flatten(),
                Dense(100, activation='relu'),
                Dropout(0.7),
                Dense(1, activation='sigmoid')])

        elif model == 'lstm':

            model = Sequential([
            Embedding(V, E,  input_length=S, weights=[self.emb], trainable=False),
            LSTM(100),
            Dense(1, activation='sigmoid')])

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        print (model.summary())
        return model


    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        '''Fits pipeline to training data.

        # Arguments:
            X_train, X_val: sequences of strings.
            y_trin, y_val: equences of ints or booleans.
            kwargs: optional arguments to keras.model.fit().

        # Returns:
            self.model: trained keras model.
        '''
        self.for_explanation = X_val
        X_train = self.tokenizer.fit_transform(X_train, y_train)
        X_val = self.tokenizer.transform(X_val)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs)
        return self.model

    def predict_proba(self, X):
        '''Predict probabilities.

        #Arguments:
            X: sequences of strings

        #Returns:
            output: Numpy ndarray of shape (len(X), 2)
        '''
        X = self.tokenizer.transform(X)
        prob_1 = self.model.predict_proba(X)
        prob_0 = 1 - prob_1
        output = np.concatenate((prob_0, prob_1), axis=1)
        return output

    def explain_one_example(self, idx=None, num_features=5, print_out=True):
        '''Explaines predictions for a single datapoint with LIME.

        If the index of the datapoint is not specified, explaines random point
        from the validation data. Optionally prints out explanation.

        # Arguments:
            idx: int, index of a datapoint in the validation data (default=None)
            num_features: int, number of explanatory features (default=5)
            print_out: boolean (default=True)

        # Returns:
            exp: lime.explanation.Explanation object
        '''
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
        '''Explains model by applying LimeTextExplainer to a specified number of
        validation datapoints and averaging up the contributions of the observed
        words.

        # Arguments:
            num_examples: int, number of datapoints.
            kwargs: optional arguments for self.explain_one_example().

        # Returns:
            mean_contrib: list of tuples (word, word`s contribution) ordered by
                absolute value of mean contribution.
        '''
        idxs = np.random.choice(self.for_explanation.index, size=num_examples, replace=False)
        contributors = defaultdict(list)

        for i in idxs:
            #Fix unicode errors to make explanations.
            while True:
                try:
                    exp = self.explain_one_example(idx=i, print_out=False, **kwargs)
                    break
                except UnicodeDecodeError:
                    i = np.random.choice(self.for_explanation.index)

            for word, weight in exp.as_list():
                contributors[word.lower()].append(weight)

        mean_contrib = []
        for key in contributors:
            mean_contrib.append((key, np.mean(contributors[key])))
        mean_contrib.sort(key=lambda x: np.abs(x[1]), reverse=True)

        return mean_contrib
