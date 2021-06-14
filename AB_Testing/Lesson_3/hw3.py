import numpy as np
import pandas as pd
from scipy.stats import norm

def estimate_sample_size(df, metric_name, effects, alpha=0.05, beta=0.2):
    """Оцениваем sample size для списка эффектов.

    df - pd.DataFrame, датафрейм с данными
    metric_name - str, название столбца с целевой метрикой
    effects - List[float], список ожидаемых эффектов. Например, [1.03] - увеличение на 3%
    alpha - float, ошибка первого рода
    beta - float, ошибка второго рода

    return - pd.DataFrame со столбцами ['effect', 'sample_size']    
    """
    # YOUR_CODE_HERE
    mu = df[metric_name].mean()
    std = df[metric_name].std(ddof=0)
    
    t_alpha = norm.ppf(1-alpha/2, loc=0, scale=1)
    t_beta = norm.ppf(1-beta, loc=0, scale=1)
    t_a_b = (t_alpha + t_beta) ** 2
    num = t_a_b * 2 * (std ** 2)
    
    sample_sizes = []
    for effect in effects:
        eps = mu * (effect - 1)
        sample_sizes.append(num / (eps ** 2))
    return pd.DataFrame({'effect': effects, 'sample_size': np.ceil(sample_sizes).astype(int)})