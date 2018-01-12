from __future__ import division, print_function
import pandas as pd
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
        df = pd.read_csv('socialmedia-disaster-tweets-DFE.csv')
        df = df[df.choose_one != "Can't Decide"]

        X = df.text
        y = df.choose_one
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        vectorizer = CountVectorizer(stop_words='english', decode_error='ignore', max_features=5000)
        X_train = vectorizer.fit_transform(X_train)

        nb = MultinomialNB()
        nb.fit(X_train, y_train)

        X_test = vectorizer.transform(X_test)
        print ('Naive Bayes model accuracy: {}'.format(nb.score(X_test, y_test)))
