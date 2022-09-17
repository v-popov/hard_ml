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

bins = np.arange(15, 70, 5)
df_users['age_bin'] = pd.cut(df_users['age'], bins, right=False, labels=bins[:-1], include_lowest=True)
df_users['stratum'] = df_users['age_bin'].astype(int) * 10 + df_users['gender'].astype(int)
df_sales = df_sales.merge(df_users, on='user_id', how='left')


historical_stratum_weights = {250: 0.230028,
 200: 0.219609,
 251: 0.173232,
 201: 0.165308,
 300: 0.083449,
 301: 0.062966,
 350: 0.022602,
 351: 0.017444,
 150: 0.006225,
 400: 0.006023,
 151: 0.004633,
 401: 0.004509,
 450: 0.001666,
 451: 0.001246,
 500: 0.000419,
 501: 0.000321,
 550: 0.000119,
 551: 0.0001,
 600: 5.7e-05,
 601: 4.4e-05}
weights = pd.Series(historical_stratum_weights)


def calc_strat_mean(df, strat_column='stratum', target_name='sales'):
    strat_mean = df.groupby(strat_column)[target_name].mean()
    return (strat_mean * weights).sum()


def calc_strat_var(df, strat_column='stratum', target_name='sales'):
    strat_var = df.groupby(strat_column)[target_name].var()
    return (strat_var * weights).sum()


def get_sub_df_with_cov(users):
    sub_df = df_sales[df_sales['user_id'].isin(users)]
    sub_df = sub_df.groupby(['user_id', 'day', 'stratum'])['sales'].sum().reset_index()

    sub_df_cov = df_sales_cov[df_sales_cov['user_id'].isin(users)]
    sub_df_cov = sub_df_cov.groupby(['user_id', 'day', 'stratum'])['sales'].sum().reset_index()

    sub_df = sub_df.merge(sub_df_cov, on=['user_id', 'day'], how='left',  # inner?
                          suffixes=('', '_cov'))
    sub_df.fillna(0., inplace=True)

    return sub_df


def test_groups(sub_df_a, sub_df_b):

    mu_a = calc_strat_mean(sub_df_a)
    mu_b = calc_strat_mean(sub_df_b)

    var_a = calc_strat_var(sub_df_a)
    var_b = calc_strat_var(sub_df_b)

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