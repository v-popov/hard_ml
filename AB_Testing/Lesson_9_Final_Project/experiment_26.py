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

# эксперимент проводился с 49 до 55 день включительно
df_sales = df_sales[df_sales['day'].isin(np.arange(49, 56))]

bins = np.arange(15, 70, 7)
df_users['age_bin'] = pd.cut(df_users['age'], bins, right=False, labels=bins[:-1], include_lowest=True)
df_users['stratum'] = df_users['age_bin'].astype(int) * 10 + df_users['gender'].astype(int)

df_sales = df_sales.merge(df_users, on='user_id', how='left')
df_sales_cov = df_sales_cov.merge(df_users, on='user_id', how='left')

historical_stratum_weights = {220: 0.366545,
 221: 0.276018,
 290: 0.122654,
 291: 0.092381,
 150: 0.057352,
 151: 0.043301,
 360: 0.019858,
 361: 0.015298,
 430: 0.003193,
 431: 0.00234,
 500: 0.000475,
 501: 0.000376,
 570: 0.00012,
 571: 8.9e-05}
weights = pd.Series(historical_stratum_weights)


def calc_strat_mean(df, strat_column, target_name):
    strat_mean = df.groupby(strat_column)[target_name].mean()
    return (strat_mean * weights).sum()


def calc_strat_var(df, strat_column, target_name):
    strat_var = df.groupby(strat_column)[target_name].var()
    return (strat_var * weights).sum()


def get_sub_df_with_cov(users):
    sub_df = df_sales[df_sales['user_id'].isin(users)]
    print(sub_df.head())
    sub_df = sub_df.groupby(['user_id', 'day', 'stratum'])['sales'].sum().reset_index()
    print(sub_df.head())

    sub_df_cov = df_sales_cov[df_sales_cov['user_id'].isin(users)]
    sub_df_cov = sub_df_cov.groupby(['user_id', 'day', 'stratum'])['sales'].sum().reset_index()

    sub_df = sub_df.merge(sub_df_cov, on=['user_id', 'day'], how='left',  # inner?
                          suffixes=('', '_cov'))
    sub_df.fillna(0., inplace=True)

    return sub_df


app = Flask(__name__)


@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    has_effect = _check_test(test)
    return jsonify(has_effect=int(has_effect))


def test_groups(user_a, user_b):
    # Cuped
    sub_df_a = get_sub_df_with_cov(user_a)
    sub_df_b = get_sub_df_with_cov(user_b)

    sales_a = sub_df_a['sales'].values
    sales_b = sub_df_b['sales'].values

    sales_a_cov = sub_df_a['sales_cov'].values
    sales_b_cov = sub_df_b['sales_cov'].values

    theta = 0.13200324053614915

    sub_df_a['sales_cuped'] = sales_a - theta * sales_a_cov
    sub_df_b['sales_cuped'] = sales_b - theta * sales_b_cov

    # Stratification
    mu_a = calc_strat_mean(sub_df_a, 'stratum', 'sales_cuped')
    mu_b = calc_strat_mean(sub_df_b, 'stratum', 'sales_cuped')

    var_a = calc_strat_var(sub_df_a, 'stratum', 'sales_cuped')
    var_b = calc_strat_var(sub_df_b, 'stratum', 'sales_cuped')

    n_a = len(sub_df_a)
    n_b = len(sub_df_a)

    _, p_val = ttest_ind_from_stats(mean1=mu_a, std1=np.sqrt(var_a), nobs1=n_a,
                                    mean2=mu_b, std2=np.sqrt(var_b), nobs2=n_b,
                                    equal_var=True, alternative="two-sided")

    return p_val


def _check_test(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    # p_val_aa = test_groups(group_a_one, group_a_two)
    # if p_val_aa < 0.05:
    #     return False

    p_val_ab = test_groups(group_a_one + group_a_two, group_b)

    return p_val_ab < 0.05