import numpy as np

def get_bernoulli_confidence_interval(values: np.array):
    """Вычисляет доверительный интервал для параметра распределения Бернулли.

    :param values: массив элементов из нулей и единиц.
    :return (left_bound, right_bound): границы доверительного интервала.
    """
    num_ones = np.sum(values)
    num_samples = len(values)
    p_hat = num_ones / num_samples
    var = p_hat * (1 - p_hat) / num_samples
    se = np.sqrt(var)

    half_interval = 1.96 * se
    left_bound = np.max((0, p_hat - half_interval))
    right_bound = np.min((1, p_hat + half_interval))
    return (left_bound, right_bound)


if __name__ == '__main__':
    print(get_bernoulli_confidence_interval(np.array([1,0,1,1,1,0])))