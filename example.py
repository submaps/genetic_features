from collections import Counter

import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ga.features_estimators import get_scores_df, timeit


def get_data(dataset_name):
    if dataset_name == 'boston':
        dataset = load_boston()
        X, y = dataset.data, dataset.target
    elif dataset_name == 'robc':
        ifile_path = '../../robc/data/train_data.csv'
        dataset = pd.read_csv(ifile_path)
        y = dataset.pop('target')
        X = dataset
        X = pd.get_dummies(X).values
    elif dataset_name == 'breast_cancer':
        dataset = load_breast_cancer()
        X, y = dataset.data, dataset.target
    return X, y

@timeit
def main():
    mode = 'classification'
    if mode == 'regression':
        dataset_name = 'boston'
        X, y = get_data(dataset_name)
        est_dict = {'lasso': Lasso(),
                    'linreg': LinearRegression()}
    elif mode == 'classification':
        dataset_name = 'breast_cancer'
        X, y = get_data(dataset_name)
        print('class balance', Counter(y))
        est_dict = {
                    # 'knn': KNeighborsClassifier(), # not support, need .features_ attirbutes
                    'logreg': LogisticRegression(),
                    'dtree': DecisionTreeClassifier(max_depth=5)
                    }
    else:
        raise Exception('invalid mode ' + mode)

    features_scores = get_scores_df(est_dict, X, y, mode)
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
# 'main'  0.30 m

# robc
#   est_name    name     mse   rmse  feat_count            time
# 0    lasso    init  104.73  10.23         147        00:00:00
# 1    lasso     RFE   43.95   6.63           2 00:01:37.186159
# 2    lasso      RF   37.82   6.15          13 00:01:58.984803
# 3    lasso  boruta  154.90  12.45          28 00:30:04.636176
# 4    lasso     gen   43.25   6.58          87 00:16:51.823936
# 'main' 50 m

# classification breast_cancer
#   est_name    name  f1_macro  feat_count            time
# 0   logreg    init      0.95          30 00:00:00.008069
# 1   logreg     RFE      0.95          26 00:00:00.492118
# 2   logreg      RF      0.95          19 00:00:00.656168
# 3   logreg  boruta      0.95          22 00:00:11.938339
# 4   logreg     gen      0.96          15 00:00:25.087129
# 5    dtree    init      0.92          30        00:00:00
# 6    dtree     RFE      0.94           3 00:00:00.611138
# 7    dtree      RF      0.93          19 00:00:00.621178
# 8    dtree  boruta      0.92          23 00:00:11.382930
# 9    dtree     gen      0.92          23 00:00:30.611924
