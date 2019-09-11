# genetic
Using simple genetic algorithm for feature selection for regression model.


```dataset_name = 'boston'
X, y = get_data(dataset_name)
est_name = 'lasso'
est_dict = {'lasso': Lasso(),
            'linreg': LinearRegression()}
est = est_dict[est_name]
features_scores = get_scores_df(est_name, est, X, y)
print(features_scores.to_string())
```

| # | est_name | name   | mse   | rmse | feat_count | time            |
|---|----------|--------|-------|------|------------|-----------------|
| 0 | lasso    | init   | 35.53 | 5.96 | 13         | 00:00:00        |
| 1 | lasso    | RFE    | 35.53 | 5.96 | 13         | 00:00:00.061016 |
| 2 | lasso    | RF     | 38.42 | 6.20 | 9          | 00:00:01.218298 |
| 3 | lasso    | boruta | 38.42 | 6.20 | 9          | 00:00:10.512721 |
| 4 | lasso    | gen    | 33.15 | 5.76 | 9          | 00:00:07.069833 |
