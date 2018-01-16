from __future__ import division, print_function

from collections import defaultdict
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from lime.lime_text import LimeTextExplainer


def baseline(X_train, y_train, X_test, y_test, max_features=5000,
             f=CountVectorizer):
    '''Trains and evaluates Naive Bayes model on the bag of words text feature
    representation.

    # Arguments:
        X_train, y_train, X_test, y_test: sequence of strings
        y: sequence of ints or booleans
        max_features: int, vocabulary size (default=5000)
        f: sklearn vectorizer (default=CountVectorizer)

    # Returns:
        vectorizer: fitted sklearn vectorizer object
        nb: fitted sklearn.naive_bayes.MultinomialNB object
    '''
    vectorizer = f(stop_words='english',
                   decode_error='ignore',
                   max_features=max_features)
    X_train = vectorizer.fit_transform(X_train)

    nb = MultinomialNB()
    nb.fit(X_train, y_train.squeeze())

    X_test = vectorizer.transform(X_test)
    print ('Naive Bayes model accuracy: {}'.format(nb.score(X_test, y_test.squeeze())))
    return vectorizer, nb

def random_forest(X_train, y_train, X_test, y_test, max_features=5000,
                  f=TfidfVectorizer):
    '''Trains and evaluates Random Forest model on the bag of words text feature
    representation.

    # Arguments:
        X_train, y_train, X_test, y_test: sequence of strings
        y: sequence of ints or booleans
        max_features: int, vocabulary size (default=5000)
        f: sklearn vectorizer (default=TfIdfVectorizer)

    # Returns:
        vectorizer: fitted sklearn vectorizer object
        rf: fitted sklearn.naive_bayes.RandomForestClassifier object
    '''
    vectorizer = f(stop_words='english',
                   decode_error='ignore',
                   max_features=max_features)
    X_train = vectorizer.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(X_train, y_train.squeeze())

    X_test = vectorizer.transform(X_test)
    print ('Random Forest model accuracy: {}'.format(rf.score(X_test, y_test.squeeze())))
    return vectorizer, rf

def explain_model(vectorizer, model, X_test, num_examples=10, num_features=5,
                  class_names=['Not relevant', 'Relevant']):
    '''Explains pipeline of sklearn vectorizer+model with LIME

    #Arguments:
        vectorizer: fitted sklearn vectorizer object
        model: fitted sklearn model
        X_test: sequence of strings
        num_examples: int, number of datapoints for explanation
        num_features: int, number of explanatory features for single example
        class_names: list of strings, class names
    # Returns:
        mean_contrib: list of tuples (word, word`s contribution) ordered by
            absolute value of mean contribution.
'''
    pipe = make_pipeline(vectorizer, model)
    idxs = np.random.choice(X_test.index, size=num_examples, replace=False)
    contributors = defaultdict(list)

    for i in idxs:
        #Fix unicode errors to make explanations.
        while True:
            try:
                explainer = LimeTextExplainer(class_names=class_names)
                exp = explainer.explain_instance(X_test[i],
                                                 pipe.predict_proba,
                                         num_features=num_features)
                break
            except UnicodeDecodeError:
                i = np.random.choice(X_test.index)

        for word, weight in exp.as_list():
            contributors[word.lower()].append(weight)

    mean_contrib = []
    for key in contributors:
        mean_contrib.append((key, np.mean(contributors[key])))
    mean_contrib.sort(key=lambda x: np.abs(x[1]), reverse=True)

    return mean_contrib

def baseline_grid_search(X_train, y_train):
    '''Performs grid search over parameters of baseline model. Returns best
    estimator.
    '''
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
    
