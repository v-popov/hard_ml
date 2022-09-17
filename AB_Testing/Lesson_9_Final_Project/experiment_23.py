import os
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_ind_from_stats
from flask import Flask, jsonify, request

# получить данные о пользователях и их покупках
df_users = pd.read_csv(os.environ['PATH_DF_USERS'])
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])

df_sales_cov = df_sales[df_sales['day'].isin(np.arange(42, 49))]
df_sales_cov['day'] += 7

# эксперимент проводился с 49 до 55 день включительно
df_sales = df_sales[df_sales['day'].isin(np.arange(49, 56))]


def calculate_theta(y_control, y_pilot, y_control_cov, y_pilot_cov) -> float:
    y = np.hstack([y_control, y_pilot])
    y_cov = np.hstack([y_control_cov, y_pilot_cov])
    covariance = np.cov(y_cov, y)[0, 1]
    variance = y_cov.var()
    theta = covariance / variance
    return theta

def linearize(df, kappa=None):
    aggregation = df.groupby(['user_id', 'day']).agg(x=('sales', 'sum'),
                                                     y=('sales', 'count'))
    if kappa is None:
        kappa = aggregation.sum().x / aggregation.sum().y

    linearized = aggregation.x - kappa * aggregation.y
    linearized = linearized.reset_index(name='sales')

    return linearized, kappa


app = Flask(__name__)

@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    has_effect = _check_test(test)
    return jsonify(has_effect=int(has_effect))

def _check_test(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b


    sales_a = df_sales[df_sales['user_id'].isin(user_a)]
    sales_b = df_sales[df_sales['user_id'].isin(user_b)]

    sales_a, kappa = linearize(sales_a)
    sales_b, _ = linearize(sales_b, kappa)


    sales_a_cov = df_sales_cov[df_sales_cov['user_id'].isin(user_a)]
    sales_b_cov = df_sales_cov[df_sales_cov['user_id'].isin(user_b)]

    sales_a_cov, kappa = linearize(sales_a_cov)
    sales_b_cov, _ = linearize(sales_b_cov, kappa)


    sales_a = sales_a.merge(sales_a_cov, on=['user_id', 'day'], how='left', suffixes=('', '_cov'))
    sales_a.fillna(0., inplace=True)
    sales_b = sales_b.merge(sales_b_cov, on=['user_id', 'day'], how='left', suffixes=('', '_cov'))
    sales_b.fillna(0., inplace=True)

    theta = calculate_theta(y_control=sales_a['sales'], y_pilot=sales_b['sales'],
                            y_control_cov=sales_a['sales_cov'], y_pilot_cov=sales_b['sales_cov'])

    sales_a_cuped = sales_a['sales'] - theta * sales_a['sales_cov']
    sales_b_cuped = sales_b['sales'] - theta * sales_b['sales_cov']

    return ttest_ind(sales_a_cuped, sales_b_cuped)[1] < 0.05
