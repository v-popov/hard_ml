import pandas as pd
import numpy as np
import joblib
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from sklearn.model_selection import RandomizedSearchCV, train_test_split, ShuffleSplit

import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

IS_RUN_CV = True

x_train = joblib.load('x.j')
print(f"X Train columns: {x_train.columns}")
x_train = x_train.values
y_train = joblib.load('y.j').values
cv = joblib.load('cv.j')

def hp_opt():
    space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
             'gamma': hp.uniform('gamma', 0.1, 9),
             'reg_alpha': hp.quniform('reg_alpha', 0, 180, 1),
             'reg_lambda': hp.uniform('reg_lambda', 0, 1),
             'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
             'subsample': hp.uniform('subsample', 0.3, 1),
             'min_child_weight': hp.quniform('min_child_weight', 0, 15, 1),
             'n_estimators': 400,
             'seed': 1
             }

    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=50,
                            trials=trials)

    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)

def objective(space):
    clf = xgb.XGBRegressor(n_jobs=15, n_estimators=300, seed=1, max_depth=int(space['max_depth']), gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree']))

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
    mse_scores = []
    for train_inds, test_inds in cv.split(x_train):
        X_train = x_train[train_inds]
        Y_train = y_train[train_inds]
        X_test = x_train[test_inds]
        Y_test = y_train[test_inds]
        evaluation = [(X_train, Y_train), (X_test, Y_test)]

        clf.fit(X_train, Y_train,
                eval_set=evaluation, eval_metric="rmse",
                early_stopping_rounds=10, verbose=False)

        pred = clf.predict(X_test)
        mse = np.mean((Y_test - pred) ** 2)
        mse_scores.append(mse)
    mean_score = np.mean(mse_scores)
    print("MEAN MSE SCORE:", mean_score, "; max-depth=", int(space['max_depth']))
    return {'loss': mean_score, 'status': STATUS_OK}



def test(x_train, y_train):
    params = {'colsample_bytree': 0.8079726723368229, 'gamma': 4.832661751357535, 'max_depth': 4, 'min_child_weight': 14, 'reg_alpha': 144, 'reg_lambda': 0.5375501521180261, 'subsample': 0.6977877918933332}
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
    mse_scores = []
    for train_inds, test_inds in cv.split(x_train):
        if params:
            model = xgb.XGBRegressor(n_jobs=15, n_estimators=400, seed=1, **params)
        else:
            model = xgb.XGBRegressor(n_jobs=15, n_estimators=400, seed=1, subsample=1.0, min_child_weight=1.6361503076509394, max_depth=3, gamma=1.0774387353144212, colsample_bytree=0.6)
        model.fit(x_train[train_inds], y_train[train_inds])
        preds = model.predict(x_train[test_inds])
        mse = np.mean((y_train[test_inds] - preds) ** 2)
        print(f"MSE: {mse}")
        mse_scores.append(mse)
    print(f"mse_scores: {mse_scores}")
    print(f"mean MSE: {np.mean(mse_scores)}")

    model.fit(x_train, y_train)
    joblib.dump(model, 'model.j')


def train_lasso(x_train, y_train):
    scaler = StandardScaler()
    for col in x_train.columns:
        print(f"{col}: min={x_train[col].min()}; max={x_train[col].max()}")
        if np.abs(x_train[col].max()) == np.inf or np.abs(x_train[col].min()) == np.inf:
            print('===ALARMA ABOVE')
    x_train = scaler.fit_transform(x_train.values)
    joblib.dump(scaler, 'scaler.j')

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
    mse_scores = []
    for train_inds, test_inds in cv.split(x_train):
        model = Lasso(random_state=1)
        model.fit(x_train[train_inds], y_train[train_inds])
        preds = model.predict(x_train[test_inds])
        mse = np.mean((y_train[test_inds] - preds) ** 2)
        print(f"MSE: {mse}")
        mse_scores.append(mse)
    print(f"mse_scores: {mse_scores}")
    print(f"mean MSE: {np.mean(mse_scores)}")

    model = Lasso(random_state=1)
    model.fit(x_train, y_train)
    joblib.dump(model, 'model_lasso.j')

if __name__ == '__main__':
    #hp_opt()
    #test(x_train, y_train)


    # x_train = joblib.load('x_train.j')
    # y_train = joblib.load('y_train.j')
    # print(f"x_train shape: {x_train.shape}")

    # train_lasso(x_train, y_train)

    if not IS_RUN_CV:
        test(x_train, y_train)

    else:
        #cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
        params = {
            'min_child_weight': np.random.uniform(0.1, 15, 20),
            'gamma': np.random.uniform(0.1, 5, 10),
            'subsample': np.random.uniform(0.4, 1, 10),
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': np.arange(3, 15)
        }
        model = xgb.XGBRegressor(n_jobs=1, n_estimators=300, seed=1)

        random_search = RandomizedSearchCV(model,
                                           param_distributions=params,
                                           n_iter=50,
                                           scoring='neg_mean_squared_error',
                                           n_jobs=15,
                                           cv=cv,
                                           verbose=3,
                                           random_state=1)

        random_search.fit(x_train, y_train)

        print(f"Best Score: {random_search.best_score_}")
        print(f"Best Parames: {random_search.best_params_}")
        joblib.dump(random_search.best_estimator_, 'model.j')



    # [CV] ALL FUNCTIONS
    # subsample = 0.6, min_child_weight = 6.575713371914831, max_depth = 9, gamma = 3.6174078848798175, colsample_bytree = 1.0
    # [CV]
    # subsample = 1.0, min_child_weight = 11.435290385326189, max_depth = 6, gamma = 0.9937263746082028, colsample_bytree = 0.6, score = -8414156679562.150, total = 5.8
    # s
    # [CV]
    # subsample = 0.6, min_child_weight = 6.575713371914831, max_depth = 9, gamma = 3.6174078848798175, colsample_bytree = 1.0

    # [CV] SUBSET FUNC
    # subsample = 0.6, min_child_weight = 3.886799194691016, max_depth = 14, gamma = 2.437924037371488, colsample_bytree = 0.8
    # [CV]
    # subsample = 0.6, min_child_weight = 8.266480598962437, max_depth = 3, gamma = 2.7162959097818615, colsample_bytree = 0.8, score = -7506327064723.357, total = 5.2
    # s
    # [CV]
    # subsample = 0.6, min_child_weight = 3.886799194691016, max_depth = 14, gamma = 2.437924037371488, colsample_bytree = 0.8

    # [CV]
    # subsample = 0.6, min_child_weight = 8.266480598962437, max_depth = 3, gamma = 2.7162959097818615, colsample_bytree = 0.8
    # [CV]
    # subsample = 1.0, min_child_weight = 13.141187633715976, max_depth = 3, gamma = 1.0173602012609022, colsample_bytree = 0.8, score = -7530472375206.274, total = 4.0
    # s
    # [CV]
    # subsample = 0.6, min_child_weight = 8.266480598962437, max_depth = 3, gamma = 2.7162959097818615, colsample_bytree = 0.8


    # [CV]
    # subsample = 0.6, min_child_weight = 6.612607886104013, max_depth = 3, gamma = 2.9187882701057153, colsample_bytree = 0.8
    # [CV]
    # subsample = 1.0, min_child_weight = 9.528456310663906, max_depth = 3, gamma = 0.5390316192840483, colsample_bytree = 0.8, score = -5696729124503.859, total = 3.6
    # s
    # [CV]
    # subsample = 0.6, min_child_weight = 6.612607886104013, max_depth = 3, gamma = 2.9187882701057153, colsample_bytree = 0.8


