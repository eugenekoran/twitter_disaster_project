# encoding: utf-8

from __future__ import division, print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from keraspipeline import KerasPipeline



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

    pipe = KerasPipeline('lstm', emb_path='full_emb.txt')
    pipe.fit(X_train, y_train, X_test, y_test, epochs=8, batch_size=64)
