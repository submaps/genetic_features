# genetic
Using simple genetic algorithm for feature selection for regression and classification models.

Regression on boston dataset
```
dataset_name = 'boston'
mode = 'regression'
X, y = get_data(dataset_name)
est_dict = {'lasso': Lasso(),
            'linreg': LinearRegression()}
features_scores = get_scores_df(est_dict, X, y, mode)
print(features_scores)
```
| # | est_name | name   | mse   | rmse | feat_count | time            |
|---|----------|--------|-------|------|------------|-----------------|
| 0 | lasso    | init   | 35.53 | 5.96 | 13         | 00:00:00        |
| 1 | lasso    | RFE    | 35.53 | 5.96 | 13         | 00:00:00.061016 |
| 2 | lasso    | RF     | 38.42 | 6.20 | 9          | 00:00:01.218298 |
| 3 | lasso    | boruta | 38.42 | 6.20 | 9          | 00:00:10.512721 |
| 4 | lasso    | gen    | 33.15 | 5.76 | 9          | 00:00:07.069833 |

Classification on breast cancer dataset
 ```
dataset_name = 'breast_cancer'
mode = 'classification'
X, y = get_data(dataset_name)
est_dict = {'logreg': LogisticRegression(),
            'dtree': DecisionTreeClassifier(max_depth=5)}
features_scores = get_scores_df(est_dict, X, y, mode)
print(features_scores)
```
| est_name | name   | f1_macro | feat_count | time            |
|----------|--------|----------|------------|-----------------|
| logreg   | init   | 0.95     | 30         | 00:00:00.008069 |
| logreg   | RFE    | 0.95     | 26         | 00:00:00.492118 |
| logreg   | RF     | 0.95     | 19         | 00:00:00.656168 |
| logreg   | boruta | 0.95     | 22         | 00:00:11.938339 |
| logreg   | gen    | 0.96     | 15         | 00:00:25.087129 |
| dtree    | init   | 0.92     | 30         | 00:00:00        |
| dtree    | RFE    | 0.94     | 3          | 00:00:00.611138 |
| dtree    | RF     | 0.93     | 19         | 00:00:00.621178 |
| dtree    | boruta | 0.92     | 23         | 00:00:11.382930 |
| dtree    | gen    | 0.92     | 23         | 00:00:30.611924 |
