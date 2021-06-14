import numpy as np
import pandas as pd


def select_stratified_groups(data, strat_columns, group_size, weights=None, seed=None):
    """Подбирает стратифицированные группы для эксперимента.

    data - pd.DataFrame, датафрейм с описанием объектов, содержит атрибуты для стратификации.
    strat_columns - List[str], список названий столбцов, по которым нужно стратифицировать.
    group_size - int, размеры групп.
    weights - dict, словарь весов страт {strat: weight}, где strat - tuple значений элементов страт,
        например, для strat_columns=['os', 'gender', 'birth_year'] будет ('ios', 'man', 1992).
        Если None, определить веса пропорционально доле страт в датафрейме data.
    seed - int, исходное состояние генератора случайных чисел для воспроизводимости
        результатов. Если None, то состояние генератора не устанавливается.

    return (data_pilot, data_control) - два датафрейма того же формата что и data
        c пилотной и контрольной группами.
    """
    # YOUR_CODE_HERE
    result = None
    for strat_column in strat_columns:
        if result is None:
            result = data[strat_column].astype(str)
        else:
            result = result + '_' + data[strat_column].astype(str)
    data['strat_key'] = result
    
    if weights is None:
        weights = data.strat_key.value_counts(normalize=True).to_dict()
    else:
        weights = {'_'.join(weight): weights[weight] for weight in weights}

    pilot = pd.DataFrame()
    control = pd.DataFrame()
    for strat_key in weights:
        strat_size_both = np.ceil(group_size * weights[strat_key]).astype(int)
        sub_df = data[data.strat_key == strat_key]
        sample = sub_df.sample(n=int(strat_size_both), random_state=seed)
        pilot = pd.concat((pilot, sample.iloc[:len(sample)//2]), ignore_index=True)
        control = pd.concat((control, sample.iloc[len(sample)//2:]), ignore_index=True)
        
    return pilot.drop(columns=['strat_key']), control.drop(columns=['strat_key'])