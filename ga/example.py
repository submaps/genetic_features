import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, LinearRegression

from ga.features_estimators import get_scores_df, timeit


def get_data(dataset_name):
    if dataset_name == 'boston':
        dataset = load_boston()
        X, y = dataset.data, dataset.target
    elif dataset_name == 'robc':
        ifile_path = r'C:/HOME/robc/data/train_data.csv'
        dataset = pd.read_csv(ifile_path)
        y = dataset.pop('target')
        X = dataset
        X = pd.get_dummies(X).values
    return X, y

@timeit
def main():
    dataset_name = 'boston'
    X, y = get_data(dataset_name)

    est_name = 'lasso'
    est_dict = {'lasso': Lasso(),
                'linreg': LinearRegression()}

    est = est_dict[est_name]
    features_scores = get_scores_df(est_name, est, X, y)
    print(features_scores.to_string())


if __name__ == '__main__':
    main()

# boston
#   est_name    name    mse  rmse  feat_count            time
# 0    lasso    init  35.53  5.96          13        00:00:00
# 1    lasso     RFE  35.53  5.96          13 00:00:00.058021
# 2    lasso      RF  38.42  6.20           9 00:00:01.219316
# 3    lasso  boruta  38.42  6.20           9 00:00:10.591762
# 4    lasso     gen  33.15  5.76           9 00:00:06.891801

# robc
#   est_name  name     mse       rmse  feat_count            time
# 0    lasso  init  104.73  10.233768         147        00:00:00
# 1    lasso   RFE   43.95   6.629480           2 00:01:36.677027
# 2    lasso    RF   37.82   6.149797          13 00:01:59.370903
