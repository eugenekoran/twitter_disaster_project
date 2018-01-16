import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer

"""
Utility functions
"""

def create_emb(filename, data, vocab_size, **kwargs):
    '''Create embeddings matrix for text data from pre-trained word vectors.

    # Arguments:
        filename: path to pretrained vectors in word2vec format.
        data: sequence of strings.
        vocab_size: int, vocabulary size.
        kwargs: arguments for keras.preprocessing.text.Tokenizer.

    # Returns:
        emb: np.ndarray, embeddings matrix.
    '''
    w2v = KeyedVectors.load_word2vec_format(filename, binary=False)

    tokenizer = Tokenizer(**kwargs)
    tokenizer.fit_on_texts(data)
    word2idx = tokenizer.word_index

    idx2word = {word2idx[key]:key for key in word2idx}

    n_fact = w2v.vector_size
    emb = np.zeros((vocab_size, n_fact))

    for i in range(1,len(emb)):
        word = idx2word[i]
        if word in w2v.vocab:
            emb[i] = w2v[word]
    return emb

def plot_explanation(mean_contrib, n):
    '''Plot top word`s contribution for predicting class 'Relevant'

    Example:

    plot_explanation(mean_contrib, 15)
    plt.savefig('contrib.png')
    plt.show()

    # Arguments:
        mean_contrib: list of tuples (str, float)
        n: int, number of contributing words to plot
    '''
    contribs, weights = zip(*mean_contrib)
    pos = range(n)
    plt.barh(pos, weights[:n][::-1])
    plt.yticks(pos, contribs[:n][::-1])
    plt.title("Average word`s contribution for class 'Relevant'")
    plt.xlabel('Weights')
    plt.tight_layout()
