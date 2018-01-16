import cPickle as pickle

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM


def load_production_model():
    vocab_size = 19842
    seq_len=33
    with open('full_emb.pkl', 'r') as f:
        emb = pickle.load(f)

    model = Sequential([
        (Embedding(vocab_size, 50,  input_length=seq_len, weights=[emb], trainable=False)),
        (LSTM(100, go_backwards=True)),
        (Dense(1, activation='sigmoid'))])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return vocab_size, seq_len, model
