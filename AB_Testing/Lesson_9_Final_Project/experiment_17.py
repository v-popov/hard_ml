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

num_buckets = 100

# эксперимент проводился с 49 до 55 день включительно
df_sales = df_sales[df_sales['day'].isin(np.arange(49, 56))]
df_sales = df_sales.groupby(['user_id', 'day'])['sales'].sum().reset_index()
df_sales['user_bucket'] = df_sales['user_id'] % num_buckets


app = Flask(__name__)


@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    has_effect = _check_test(test)
    return jsonify(has_effect=int(has_effect))


def get_sub_vals(users):
    sub_df = df_sales[df_sales['user_id'].isin(users)]
    return sub_df.groupby('user_bucket')['sales'].sum().values


def _check_test(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    sales_a = get_sub_vals(user_a)
    sales_b = get_sub_vals(user_b)

    return ttest_ind(sales_a, sales_b)[1] < 0.05