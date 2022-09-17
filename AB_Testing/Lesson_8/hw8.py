import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm

class SequentialTester:
    def __init__(
        self, metric_name, time_column_name,
        alpha, beta, pdf_one, pdf_two
    ):
        """Создаём класс для проверки гипотезы о равенстве средних тестом Вальда.

        Предполагается, что среднее значение метрики у распределения альтернативной
        гипотезы с плотность pdf_two больше.

        :param metric_name: str, название стобца со значениями измерений.
        :param time_column_name: str, названия столбца с датой и временем измерения.
        :param alpha: float, допустимая ошибка первого рода.
        :param beta: float, допустимая ошибка второго рода.
        :param pdf_one: function, функция плотности распределения метрики при H0.
        :param pdf_two: function, функция плотности распределения метрики при H1.
        """
        self.metric_name = metric_name
        self.time_column_name = time_column_name
        self.alpha = alpha
        self.beta = beta
        self.pdf_one = pdf_one
        self.pdf_two = pdf_two
        # YOUR_CODE_HERE
        self.data_control = pd.DataFrame()
        self.data_pilot = pd.DataFrame()

    
    def _run_test(self, data_control, data_pilot):
        length = len(data_control)
        
        lower_bound = np.log(self.beta / (1 - self.alpha))
        upper_bound = np.log((1 - self.beta) / self.alpha)

        delta_data = data_pilot[self.metric_name] - data_control[self.metric_name]

        pdf_one_values = self.pdf_one(delta_data)
        pdf_two_values = self.pdf_two(delta_data)

        z = np.cumsum(np.log(pdf_two_values / pdf_one_values))

        indexes_lower = np.arange(length)[z < lower_bound]
        indexes_upper = np.arange(length)[z > upper_bound]
        first_index_lower = indexes_lower[0] if len(indexes_lower) > 0 else length + 1
        first_index_upper = indexes_upper[0] if len(indexes_upper) > 0 else length + 1

        if first_index_lower < first_index_upper:
            return 0., first_index_lower + 1
        elif first_index_lower > first_index_upper:
            return 1., first_index_upper + 1
        else:
            return 0.5, length
    

    def run_test(self, data_control, data_pilot):
        """Запускаем новый тест, проверяет гипотезу о равенстве средних.
        
        :param data_control: pd.DataFrame, данные контрольной группы.
        :param data_pilot: pd.DataFrame, данные пилотной группы.
        
        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных 
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        # YOUR_CODE_HERE
#         data_control_sorted = data_control.sort_values(by=self.time_column_name).reset_index(drop=True)
#         data_pilot_sorted = data_pilot.sort_values(by=self.time_column_name).reset_index(drop=True)

        self.data_control = pd.concat((self.data_control, data_control), axis=0, ignore_index=True)
        self.data_pilot = pd.concat((self.data_pilot, data_pilot), axis=0, ignore_index=True)
        
        return self._run_test(data_control, data_pilot)
        

    def add_data(self, data_control, data_pilot):
        """Добавляет новые данные, проверяет гипотезу о равенстве средних.
        
        Гарантируется, что данные новые и не дублируют ранее добавленные.
        
        :param data_control: pd.DataFrame, новые данные контрольной группы.
        :param data_pilot: pd.DataFrame, новые данные пилотной группы.
        
        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных 
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        # YOUR_CODE_HERE
#         data_control_new = data_control.sort_values(by=self.time_column_name)
#         data_pilot_new = data_pilot.sort_values(by=self.time_column_name)
        
        self.data_control = pd.concat((self.data_control, data_control), axis=0, ignore_index=True)
        self.data_pilot = pd.concat((self.data_pilot, data_pilot), axis=0, ignore_index=True)
        return self._run_test(self.data_control, self.data_pilot)