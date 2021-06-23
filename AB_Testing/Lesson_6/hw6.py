import numpy as np
import pandas as pd

def calculate_linearized_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name, kappa=None
):
    """Вычисляет значение линеаризованной метрики для списка пользователей в определённый период.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    period - dict, словарь с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит в
        полуинтервал, а дата окончания нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики
    kappa - float, коэффициент в функции линеаризации.
        Если None, то посчитать как ratio метрику по имеющимся данным.

    return - pd.DataFrame, со столбцами [user_id_name, metric_name], кол-во строк должно быть равно
        кол-ву элементов в списке list_user_id.
    """
    # YOUR_CODE_HERE
    date_filter = (period['begin'] <= df[date_name]) & (df[date_name] < period['end'])
    sub_df = df[date_filter]
    
    sub_df = sub_df.query(f"{user_id_name} in @list_user_id")
    
    aggregation = sub_df.groupby(user_id_name).agg(x = (value_name, 'sum'),
                                                   y = (value_name, 'count'))
    if kappa is None:
        kappa = aggregation.sum().x / aggregation.sum().y
    
    linearized = aggregation.x - kappa * aggregation.y
    
    ans = pd.DataFrame({user_id_name: aggregation.index, metric_name: linearized})
    
    remaining_user_ids = np.array(list_user_id)[np.isin(list_user_id, ans[user_id_name], invert=True)]
    ans = pd.concat((ans, pd.DataFrame({user_id_name: remaining_user_ids, metric_name: 0.})))
    
    return ans.reset_index(drop=True)