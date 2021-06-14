import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_first_type_error(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибку первого рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.

    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел.

    return - float, ошибка первого рода
    """

    pilot_metrics = df_pilot_group[metric_name]
    control_metrics = df_control_group[metric_name]
    equal_var = np.isclose(np.std(pilot_metrics), np.std(control_metrics), rtol=0.03)

    np.random.seed(seed)
    type_1_errors_count = 0

    for _ in range(n_iter):
        pilot_b = np.random.choice(pilot_metrics, replace=True, size=len(pilot_metrics))
        control_b = np.random.choice(control_metrics, replace=True, size=len(control_metrics))

        _, p_val = ttest_ind(pilot_b, control_b, equal_var=equal_var)
        type_1_errors_count += p_val < alpha

    return type_1_errors_count / n_iter