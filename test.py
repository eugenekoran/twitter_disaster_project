from keras.preprocessing.text import Tokenizer
import cPickle as pickle

with open('tokenizer.pkl', 'r') as f:
    tk = pickle.load(f)
