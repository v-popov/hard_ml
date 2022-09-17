import os
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from flask import Flask, jsonify, request

# получить данные о пользователях и их покупках
df_users = pd.read_csv(os.environ['PATH_DF_USERS'])
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])

# эксперимент проводился с 49 до 55 день включительно
df_sales = df_sales[
    df_sales['day'].isin(np.arange(49, 56))
]

bins = np.arange(0., 150, 5)
bins[-1] = np.inf
df_users['age_bin'] = pd.cut(df_users.age, bins, right=False, labels=bins[:-1])
df_users['stratum'] = df_users['age_bin'].astype(float) * 100 + df_users['gender']
weights = df_users['stratum'].value_counts(normalize=True)


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
    df_sales = df_sales.groupby(['user_id', 'day'])['sales'].sum().reset_index()
    df_sales = df_sales.merge(df_users, on='user_id', how='left')
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

    sales_a = df_sales[df_sales['user_id'].isin(user_a)]#['sales'].values
    sales_b = df_sales[df_sales['user_id'].isin(user_b)]#['sales'].values

    mu_a = calc_strat_mean(sales_a, 'stratum', 'sales')
    mu_b = calc_strat_mean(sales_b, 'stratum', 'sales')

    var_a = calc_strat_var(sales_a, 'stratum', 'sales')
    var_b = calc_strat_var(sales_b, 'stratum', 'sales')

    n_a = len(sales_a)
    n_b = len(sales_b)

    _, p_val = ttest_ind_from_stats(mean1=mu_a, std1=np.sqrt(var_a), nobs1=n_a,
                                    mean2=mu_b, std2=np.sqrt(var_b), nobs2=n_b,
                                    equal_var=True, alternative="two-sided")

    return p_val < 0.05
