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
        new_weights = {}
        if isinstance(list(weights.keys())[0], tuple):
            for weight in weights:
                key = ''
                for value in weight:
                    key = f'{key}_{value}'
                new_weights[key[1:]] = weights[weight]
        else:
            for weight in weights:
                new_weights[str(weight)] = weights[weight]
        weights = new_weights
    
    weights = dict(sorted(weights.items(), key=lambda item: item[1]))
    pilot = pd.DataFrame()
    control = pd.DataFrame()
    sizes_count = 0
    for i, k in enumerate(weights):
        strat_size_both = 2 * np.ceil(group_size * weights[k]).astype(int)
        sizes_count += strat_size_both
        if i == len(weights)-1 and sizes_count != group_size * 2:
            print(f'Updating original strat_size_both of {strat_size_both}')
            strat_size_both += group_size * 2 - sizes_count
        sub_df = data[data['strat_key'] == k]
        sample = sub_df.sample(n=int(strat_size_both), random_state=seed)
        print(f'sample shape for stratum {k}: {sample.shape}')
        pilot = pd.concat((pilot, sample.iloc[:len(sample) // 2]), ignore_index=True)
        control = pd.concat((control, sample.iloc[len(sample) // 2:]), ignore_index=True)
        
    data.drop(columns=['strat_key'], inplace=True)
    pilot.drop(columns=['strat_key'], inplace=True)
    control.drop(columns=['strat_key'], inplace=True)
    
    return (pilot, control)