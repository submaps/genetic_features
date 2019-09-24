import unittest

from nltk import DecisionTreeClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

from example import get_data
from ga.features_estimators import get_scores_df


class TestGA(unittest.TestCase):
    def test_regression(self):
        dataset_name = 'boston'
        mode = 'regression'
        X, y = get_data(dataset_name)
        est_dict = {'lasso': Lasso(),
                    'linreg': LinearRegression()}
        features_scores = get_scores_df(est_dict, X, y, mode)
        print(features_scores)

    def test_classification(self):
        dataset_name = 'breast_cancer'
        mode = 'classification'
        X, y = get_data(dataset_name)
        est_dict = {'logreg': LogisticRegression(),
                    'dtree': DecisionTreeClassifier(max_depth=5)}
        features_scores = get_scores_df(est_dict, X, y, mode)
        print(features_scores)
