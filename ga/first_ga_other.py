import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

from kriging_utils import timeit

SEED = 2018
random.seed(SEED)
np.random.seed(SEED)

import warnings
warnings.filterwarnings("ignore")


def get_data(dataset_name):
    if dataset_name == 'boston':
        dataset = load_boston()
        X, y = dataset.data, dataset.target
        # features = dataset.feature_names
    elif dataset_name == 'robc':
        ifile_path = r'C:\HOME\robocats\data\train_data.csv'
        dataset = pd.read_csv(ifile_path)
        y = dataset.pop('target')
        X = dataset
        X = pd.get_dummies(X).values
        # print('X shape:', X.shape)
    return X, y


@timeit
def main():
    dataset_name = 'robc'
    X, y = get_data(dataset_name)

    # ==============================================================================
    # CV MSE before feature selection
    # ==============================================================================
    est = LinearRegression()
    score = -1.0 * cross_val_score(est, X, y, cv=5, scoring="neg_mean_squared_error")
    print("CV MSE before feature selection: {:.2f}".format(np.mean(score)))

    # ==============================================================================
    # CV MSE after feature selection: RFE
    # ==============================================================================
    rfe = RFECV(est, cv=5, scoring="neg_mean_squared_error")
    rfe.fit(X, y)
    score = -1.0 * cross_val_score(est, X[:, rfe.support_], y, cv=5, scoring="neg_mean_squared_error")
    print("CV MSE after RFE feature selection: {:.2f}".format(np.mean(score)))

    # ==============================================================================
    # CV MSE after feature selection: Feature Importance
    # ==============================================================================
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED)
    rf.fit(X, y)
    support = rf.feature_importances_ > 0.01
    score = -1.0 * cross_val_score(est, X[:, support], y, cv=5, scoring="neg_mean_squared_error")
    print("CV MSE after Feature Importance feature selection: {:.2f}".format(np.mean(score)))

    # ==============================================================================
    # CV MSE after feature selection: Boruta
    # ==============================================================================
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED)
    boruta = BorutaPy(rf, n_estimators='auto')
    boruta.fit(X, y)
    score = -1.0 * cross_val_score(est, X[:, boruta.support_], y, cv=5, scoring="neg_mean_squared_error")
    print("CV MSE after Boruta feature selection: {:.2f}".format(np.mean(score)))

    # lasso ?
    # score = -1.0 * cross_val_score(est, X[:, boruta.support_], y, cv=5, scoring="neg_mean_squared_error")


if __name__ == '__main__':
    main()

# boston dataset
# CV MSE before feature selection: 37.13
# CV MSE after RFE feature selection: 33.21
# CV MSE after Feature Importance feature selection: 35.49
# CV MSE after Boruta feature selection: 35.49
# 'main'  0.17 m

# robc
# X shape: (3263, 147)
# CV MSE before feature selection: 503377.13
# CV MSE after RFE feature selection: 66.75
# CV MSE after Feature Importance feature selection: 170.58
# CV MSE after Boruta feature selection: 69543949.36
# 'main'  32.40 m
