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


def get_test(df_with_user_id, size):
    ids = df_with_user_id['user_id'].sample(n=size * 3, replace=False).values
    group_a_one = ids[:size]
    group_a_two = ids[size:2 * size]
    group_b = ids[-size:]
    return {'group_a_one': list(group_a_one), 'group_a_two': list(group_a_two), 'group_b': list(group_b)}


def get_combo(df_sales_filtered, df_users, test_group):
    group = df_sales_filtered[df_sales_filtered.user_id.isin(test_group)]
    combo = group.merge(df_users, on='user_id', how='left')
    return combo


def run_experiment(i):
    df_sales = pd.read_feather('description_3/df_sales.csv')
    df_users = pd.read_feather('description_3/df_users.csv')

    start_day = np.random.randint(0, 35)
    end_day = start_day + 7
    df_sales_filtered = df_sales[df_sales['day'].isin(np.arange(start_day, end_day))]
    test = get_test(df_users, 100)

    group_a_one = get_combo(df_sales_filtered, df_users, test['group_a_one'])
    group_a_two = get_combo(df_sales_filtered, df_users, test['group_a_two'])
    group_b = get_combo(df_sales_filtered, df_users, test['group_b'])

    group_a = pd.concat((group_a_one, group_a_two), axis=0, ignore_index=True)

    group_a_one.to_feather(f'description_3/tests/group_a_one_{i}.csv')
    group_a_two.to_feather(f'description_3/tests/group_a_two_{i}.csv')
    group_b.to_feather(f'description_3/tests/group_b_{i}.csv')
    group_a.to_feather(f'description_3/tests/group_a_{i}.csv')

    return None


if __name__ == '__main__':
    num_experiments = 1000

    # p_vals = []
    # for ans in tqdm(range(num_experiments)):
    #     p_val = run_experiment()
    #     p_vals.append(p_val)

    t0 = time.time()
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=8) as p:
        for i, p_val in enumerate(p.imap_unordered(run_experiment, range(num_experiments)), start=1):
            print(f'{i}/{num_experiments}; {round((time.time()-t0) / i * (num_experiments - i))}s remaining')
