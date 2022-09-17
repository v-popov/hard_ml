import os
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from flask import Flask, jsonify, request

# получить данные о пользователях и их покупках
df_users = pd.read_csv(os.environ['PATH_DF_USERS'])
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])

app = Flask(__name__)


def calculate_theta(y, y_covariate) -> float:
    covariance = np.cov(y_covariate, y)[0, 1]
    variance = y_covariate.var()
    theta = covariance / variance
    return theta


@app.route('/ping')
def ping():
    global df_sales
    df_sales = df_sales.groupby(['user_id', 'day']).agg(sales=('sales', sum)).reset_index()

    min_day = df_sales['day'].min()
    max_day = df_sales['day'].max()
    full_days_horizon = np.arange(min_day, max_day + 1)

    all_users_sales = df_sales.user_id.unique()

    full_sales = pd.DataFrame({'user_id': np.repeat(all_users_sales, len(full_days_horizon)),
                               'day': np.tile(full_days_horizon, len(all_users_sales))})
    full_sales = full_sales.merge(df_sales, on=['user_id', 'day'], how='left').fillna(0.)

    df_sales_covariate = full_sales[full_sales['day'].isin(np.arange(42, 49))].reset_index(drop=True)
    df_sales_pilot = full_sales[full_sales['day'].isin(np.arange(49, 56))].reset_index(drop=True)
    theta = calculate_theta(df_sales_pilot.sales, df_sales_covariate.sales)
    y_cuped = df_sales_pilot.sales - theta * df_sales_covariate.sales

    df_sales_pilot.sales = y_cuped

    # эксперимент проводился с 49 до 55 день включительно
    df_sales = df_sales_pilot

    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    # json-запрос, с тремя ключам ['group_a_one', 'group_a_two', 'group_b'],
    # значения которых - список идентификаторов пользователей user_id (Dict[str, List[int]])
    # test: {'group_a_one': ['id_1', 'id_17', ...],
    #        'group_a_two': ['id_9', 'id_15', ...],
    #        'group_b':     ['id_3', 'id_82', ...]}
    has_effect = _check_test(test)
    return jsonify(has_effect=int(has_effect))


def _check_test(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']

    # AB test
    user_a = group_a_one + group_a_two
    user_b = test['group_b']

    sales_a = df_sales[
        df_sales['user_id'].isin(user_a)
    ][
        'sales'
    ].values

    sales_b = df_sales[
        df_sales['user_id'].isin(user_b)
    ][
        'sales'
    ].values

    return ttest_ind(sales_a, sales_b)[1] < 0.05
