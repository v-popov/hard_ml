import numpy as np
import pandas as pd


def calculate_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name
):
    """Вычисляет значение метрики для списка пользователей в определённый период.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    period - dict, словарь с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит нужный
        полуинтервал, а дата окончание нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами [user_id_name, metric_name], кол-во строк должно быть равно
        кол-ву элементов в списке list_user_id.
    """
    # YOUR_CODE_HERE 
    date_filter = (period['begin'] <= df[date_name]) & (df[date_name] < period['end'])
    sub_df = df[date_filter]
    
    sub_df = sub_df.query(f"{user_id_name} in @list_user_id")
    
    ans = sub_df.groupby(user_id_name)[value_name].sum().reset_index()#.to_dict()
    ans.columns = [user_id_name, metric_name]
    remaining_user_ids = np.array(list_user_id)[np.isin(list_user_id, ans[user_id_name], invert=True)]
    ans = pd.concat((ans, pd.DataFrame({user_id_name: remaining_user_ids, metric_name: 0.})))

    return ans.reset_index(drop=True)

def calculate_metric_cuped(
    df, value_name, user_id_name, list_user_id, date_name, periods, metric_name
):
    """Вычисляет метрики во время пилота, коварианту и преобразованную метрику cuped.
    
    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    periods - dict, словарь с датами начала и конца периода пилота и препилота.
        Пример, {
            'prepilot': {'begin': '2020-01-01', 'end': '2020-01-08'},
            'pilot': {'begin': '2020-01-08', 'end': '2020-01-15'}
        }.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами
        [user_id_name, metric_name, f'{metric_name}_prepilot', f'{metric_name}_cuped'],
        кол-во строк должно быть равно кол-ву элементов в списке list_user_id.
    """
    # YOUR_CODE_HERE
    df_prepilot = calculate_metric(df, value_name, user_id_name, list_user_id, date_name, 
                                   periods['prepilot'], f'{metric_name}_prepilot')
    df_pilot = calculate_metric(df, value_name, user_id_name, list_user_id, date_name, 
                                periods['pilot'], metric_name)
    
    df_combo = df_pilot.merge(df_prepilot, on=user_id_name)
    df_combo.fillna(0., inplace=True)
    
    cov = np.cov([df_combo[f'{metric_name}_prepilot'], df_combo[metric_name]])[0, 1]
    theta = cov / np.var(df_combo[f'{metric_name}_prepilot'])
    
    df_combo[f'{metric_name}_cuped'] = df_combo[metric_name] - theta * df_combo[f'{metric_name}_prepilot']
    return df_combo