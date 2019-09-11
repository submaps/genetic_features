import random
import numpy as np
import time

import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

from ga.GeneticSelector import GeneticSelector

SEED = 2018
random.seed(SEED)
np.random.seed(SEED)

import warnings

warnings.filterwarnings("ignore")


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f m' % (method.__name__, int(te - ts) / 60))
        return result

    return timed


def get_cv_score(est, X_features, y):
    return -1.0 * cross_val_score(est, X_features, y, cv=5, scoring="neg_mean_squared_error")


def get_RFE_features(est, X, y):
    # recursive feature estimator
    rfe = RFECV(est, cv=5, scoring="neg_mean_squared_error")
    rfe.fit(X, y)
    X_features = X[:, rfe.support_]
    return X_features


def get_RF_feature_importance(est, X, y):
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED)
    rf.fit(X, y)
    support = rf.feature_importances_ > 0.01
    X_features = X[:, support]
    return X_features


def get_boruta_features(est, X, y):
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED)
    boruta = BorutaPy(rf, n_estimators='auto')
    boruta.fit(X, y)
    X_features = X[:, boruta.support_]
    return X_features


def get_genetic_features(est, X, y, need_plot_scores=False):
    sel = GeneticSelector(estimator=est,
                          n_gen=7, size=200, n_best=40, n_rand=40,
                          n_children=5, mutation_rate=0.05)
    sel.fit(X, y)
    if need_plot_scores:
        sel.plot_scores()
    X_features = X[:, sel.support_]
    return X_features


def get_scores_df(est_name, est, X, y, features_selectors=('init', 'RFE', 'RF', 'boruta', 'gen')):
    features_selectors_dict = {
        'init': lambda est, X, y: X,
        'RFE': get_RFE_features,
        'RF': get_RF_feature_importance,
        'boruta': get_boruta_features,
        'gen': get_genetic_features
    }
    selector_scores = []
    for feat_selector in features_selectors:
        selector_start = pd.Timestamp.now()
        X_features = features_selectors_dict[feat_selector](est, X, y)
        elapsed_time = pd.Timestamp.now() - selector_start
        feat_count = X_features.shape[1]
        cv_scores = get_cv_score(est, X_features, y)
        mse = round(np.mean(cv_scores), 2)
        rmse = round(mse**0.5, 2)
        selector_score = {'est_name': est_name,
                          'name': feat_selector,
                          'mse': mse,
                          'rmse': rmse,
                          'feat_count': feat_count,
                          'time': elapsed_time}
        selector_scores.append(selector_score)
        print(feat_selector, mse, feat_count, elapsed_time)
    return pd.DataFrame(selector_scores)
