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

# эксперимент проводился с 49 до 55 день включительно
df_sales = df_sales[
    df_sales['day'].isin(np.arange(49, 56))
]


weights = df_users.gender.value_counts(normalize=True)


def calc_strat_mean(df, strat_column, target_name):
    strat_mean = df.groupby(strat_column)[target_name].mean()
    return (strat_mean * weights).sum()


def calc_strat_var(df, strat_column, target_name):
    strat_var = df.groupby(strat_column)[target_name].var()
    return (strat_var * weights).sum()


app = Flask(__name__)

@app.route('/ping')
def ping():
    global df_sales
    df_sales = df_sales.groupby('user_id')['sales'].sum().reset_index()
    df_sales = df_sales.merge(df_users, on='user_id', how='left')
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

    sales_a = df_sales[df_sales['user_id'].isin(user_a)]#['sales'].values
    sales_b = df_sales[df_sales['user_id'].isin(user_b)]#['sales'].values

    mu_a = calc_strat_mean(sales_a, 'gender', 'sales')
    mu_b = calc_strat_mean(sales_b, 'gender', 'sales')

    var_a = calc_strat_var(sales_a, 'gender', 'sales')
    var_b = calc_strat_var(sales_b, 'gender', 'sales')

    delta_mean_strat = mu_b - mu_a
    std_mean_strat = (var_b / len(sales_b) + var_a / len(sales_a)) ** 0.5

    no_effect = delta_mean_strat - 1.96 * std_mean_strat <= 0 <= delta_mean_strat + 1.96 * std_mean_strat

    return not no_effect
