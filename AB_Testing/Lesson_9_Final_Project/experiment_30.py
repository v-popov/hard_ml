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

df_sales_cov = df_sales[df_sales['day'].isin(np.arange(35, 42))]
df_sales_cov['day'] += 14
df_sales_cov = df_sales_cov.groupby(['user_id'])['sales'].sum().reset_index()

# эксперимент проводился с 49 до 55 день включительно
df_sales = df_sales[df_sales['day'].isin(np.arange(49, 56))]
df_sales = df_sales.groupby(['user_id'])['sales'].sum().reset_index()

df_sales = df_sales.merge(df_sales_cov, on='user_id', how='outer', suffixes=('', '_cov')).fillna(0.)

theta = 0.8936254474692923
df_sales['sales'] = df_sales['sales'] - theta * df_sales['sales_cov']


def test_groups(sub_df_a, sub_df_b):
    sales_a = sub_df_a['sales'].values
    sales_b = sub_df_b['sales'].values

    mu_a = np.mean(sales_a)
    mu_b = np.mean(sales_b)

    var_a = np.var(sales_a)
    var_b = np.var(sales_b)

    n_a = len(sub_df_a)
    n_b = len(sub_df_b)

    _, p_val = ttest_ind_from_stats(mean1=mu_a, std1=np.sqrt(var_a), nobs1=n_a,
                                    mean2=mu_b, std2=np.sqrt(var_b), nobs2=n_b,
                                    equal_var=True, alternative="two-sided")

    return p_val


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

    sales_a1 = df_sales[df_sales['user_id'].isin(group_a_one)]
    sales_a2 = df_sales[df_sales['user_id'].isin(group_a_two)]
    p_val_aa = test_groups(sales_a1, sales_a2)
    if p_val_aa < 0.05:
        return False

    user_a = group_a_one + group_a_two
    user_b = group_b

    sales_a = df_sales[df_sales['user_id'].isin(user_a)]
    sales_b = df_sales[df_sales['user_id'].isin(user_b)]

    p_val_ab = test_groups(sales_a, sales_b)


    return p_val_ab < 0.05