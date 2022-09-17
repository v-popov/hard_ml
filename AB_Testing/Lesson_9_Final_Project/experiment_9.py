import os
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t
from flask import Flask, jsonify, request

# получить данные о пользователях и их покупках
df_users = pd.read_csv(os.environ['PATH_DF_USERS'])
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])

# эксперимент проводился с 49 до 55 день включительно
df_sales = df_sales[
    df_sales['day'].isin(np.arange(49, 56))
]


app = Flask(__name__)

@app.route('/ping')
def ping():
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
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

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

    mu_a = sales_a.mean()
    mu_b = sales_b.mean()
    size_a = len(sales_a)
    size_b = len(sales_b)

    var_a = sales_a.var(ddof=1)
    var_b = sales_b.var(ddof=1)

    var_a_norm = var_a / size_a
    var_b_norm = var_b / size_b

    t_stat = (mu_b - mu_a) / np.sqrt(var_a_norm + var_b_norm)

    dof = size_a + size_b - 2

    p = (1 - t.cdf(t_stat, df=dof)) * 2

    return p < 0.05
