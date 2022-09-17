from tqdm import * #tqdm
import multiprocessing
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats, ttest_ind
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
import time

# # эксперимент проводился с 49 до 55 день включительно
# df_sales = df_sales[df_sales['day'].isin(np.arange(49, 56))]
# df_sales = df_sales.groupby(['user_id', 'day'])['sales'].sum().reset_index()
#
# bins = np.arange(15, 70, 5)
# df_users['age_bin'] = pd.cut(df_users['age'], bins, right=False, labels=bins[:-1], include_lowest=True)
# df_users['stratum'] = df_users['age_bin'].astype(int) * 10 + df_users['gender'].astype(int)
#
# df_sales = df_sales.merge(df_sales_cov, on=['user_id', 'day'], how='left', suffixes=['', '_cov'])
# df_sales.fillna(0., inplace=True)
# df_sales = df_sales.merge(df_users, on='user_id', how='left')
#
# #df_sales_cov = df_sales_cov.merge(df_users, on='user_id', how='left')
#
# historical_stratum_weights = {250: 0.230028,
#  200: 0.219609,
#  251: 0.173232,
#  201: 0.165308,
#  300: 0.083449,
#  301: 0.062966,
#  350: 0.022602,
#  351: 0.017444,
#  150: 0.006225,
#  400: 0.006023,
#  151: 0.004633,
#  401: 0.004509,
#  450: 0.001666,
#  451: 0.001246,
#  500: 0.000419,
#  501: 0.000321,
#  550: 0.000119,
#  551: 0.0001,
#  600: 5.7e-05,
#  601: 4.4e-05}
# weights = pd.Series(historical_stratum_weights)


def calc_strat_mean(df, strat_column, target_name):
    strat_mean = df.groupby(strat_column)[target_name].mean()
    return (strat_mean * weights).sum()


def calc_strat_var(df, strat_column, target_name):
    strat_var = df.groupby(strat_column)[target_name].var()
    return (strat_var * weights).sum()


def plot_pvalue_ecdf(pvalues, title=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if title:
        plt.suptitle(title)

    sns.histplot(pvalues, ax=ax1, bins=20, stat='density')
    ax1.plot([0, 1], [1, 1], 'k--')
    ax1.set_xlabel('p-value')
    ax1.set_xticks(np.arange(0, 1.1, 0.2), minor=False)
    ax1.set_xlim((0, 1))

    sns.ecdfplot(pvalues, ax=ax2)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('p-value')
    ax2.grid()
    ax2.set_xticks(np.arange(0, 1.1, 0.2), minor=False)
    ax2.set_yticks(np.arange(0, 1.1, 0.2), minor=False)
    ax2.plot([0, 1], [0, 1], 'r--')
    ax2.set_xlim((0, 1))

    plt.savefig('description_3/qwe.png')


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


def run_experiment(i):
    #df_users = pd.read_feather('description_3/df_users.csv')
    group_a = pd.read_feather(f'description_3/tests/group_a_{i}.csv')
    group_b = pd.read_feather(f'description_3/tests/group_b_{i}.csv')

    group_a = group_a.groupby(['user_id'])['sales'].sum().reset_index()
    group_b = group_b.groupby(['user_id'])['sales'].sum().reset_index()
    # quantile = np.percentile(df_sales['sales'], 99)
    # group_a = group_a[group_a['sales'] < np.percentile(group_a['sales'], 95)]
    # group_b = group_b[group_b['sales'] < np.percentile(group_b['sales'], 95)]

    return test_groups(group_a, group_b)


if __name__ == '__main__':
    num_experiments = 1000

    p_vals = []
    for i, ans in enumerate(tqdm(range(num_experiments))):
        p_val = run_experiment(i)
        p_vals.append(p_val)

    # p_vals = []
    # t0 = time.time()
    # multiprocessing.set_start_method('spawn', force=True)
    # with multiprocessing.Pool(processes=8) as p:
    #     for i, p_val in enumerate(p.imap_unordered(run_experiment, range(num_experiments)), start=1):
    #         print(f'{i}/{num_experiments}; {round((time.time()-t0) / i * (num_experiments - i))}s remaining')
    #         p_vals.append(p_val)

    joblib.dump(p_vals, 'description_3/p_vals.j')
    plot_pvalue_ecdf(p_vals)
