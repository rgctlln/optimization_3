import time
import tracemalloc

import numpy as np
import pandas
from IPython.display import display


def required_step_defined(method):
    """
    Обязательный декоратор для всех методов, кроме инициализирующих.

    Проверяет заданность шага.
    :param method:
    :return:
    """

    def wrapper(self, *args, **kwargs):
        if self.step is None:
            raise Exception("The step is not defined, call the step setup")
        return method(self, *args, **kwargs)

    return wrapper


class BackTrace:

    def __init__(self, parameters: pandas.Series, points: pandas.DataFrame, batch_size: int = 1):
        self.start_step = None
        self.step = None  # требуется вызов установки шага

        self.parameters: pandas.Series = parameters
        self.points = points

        self.choose_step_mode = None
        self.exponential_decay = None

        self.cnt_iterations = 0

        self.full_output = False

        self.polynomial_decay_alpha = None
        self.polynomial_decay_beta = None

        self.dichotomy_range = None

        self.batch_size = batch_size
        return

    def set_full_output(self):
        self.full_output = True
        return self

    def set_constant_step(self, constant_step: int | float):
        """
        Устанавливает шаг как константу

        :param constant_step:
        :return:
        """

        self.choose_step_mode = "constant"
        self.step = constant_step
        return self

    def set_piecewise_constant(self, start_step: int | float = 1):
        """
        Устанавливаем шаг как кусочно-постоянную функцию

        :return:
        """
        self.choose_step_mode = "piecewise_constant"
        self.step = start_step
        return self

    def set_exponential_decay(self, exponential_decay: float, start_step: int | float):
        """
        Устанавливаем шаг как кусочно-постоянную функцию

        :return:
        """
        self.choose_step_mode = "exponential_decay"
        self.exponential_decay = exponential_decay
        self.step = start_step
        self.start_step = start_step
        return self

    # добавил Кирюха(то есть я) надо проверить на валидность
    def set_polynomial_decay(self, alpha: float = 0.5, beta: float = 1.0, start_step: float = 1.0):
        """
        Устанавливает полиномиальное затухание: h(k) = h0 / (βk + 1)^α

        :param alpha: показатель степени α
        :param beta: множитель при k
        :param start_step: начальный шаг h0
        """
        self.choose_step_mode = "polynomial_decay"
        self.polynomial_decay_alpha = alpha
        self.polynomial_decay_beta = beta
        self.step = start_step
        self.start_step = start_step
        return self

    def set_dichotomy_step(self, a=1e-5, b=1.0):
        """
        Устанавливаем метод выбора шага через дихотомию

        :param a: левая граница отрезка
        :param b: правая граница отрезка
        """
        self.choose_step_mode = "dichotomy"
        self.dichotomy_range = (a, b)
        self.step = 1  # нужно что-то, чтобы не ругался декоратор
        return self

    def set_golden_section_step(self, a=1e-5, b=1.0):
        """
        Устанавливаем метод выбора шага через метод золотого сечения

        :param a: левая граница отрезка
        :param b: правая граница отрезка
        """
        self.choose_step_mode = "golden_section"
        self.dichotomy_range = (a, b)  # используем то же поле
        self.step = 1  # просто чтобы декоратор не ругался
        return self

    @required_step_defined
    def start_back_trace(self) -> pandas.Series:
        """
        Запуск градиентного спуска
        :return:
        """
        tracemalloc.start()
        start_time = time.time()
        epsilon = 1e-6
        history_last_norma = -1
        cnt_history_last_norma = 0
        self.history = []

        self.parameters = self.parameters.astype(float)

        if self.batch_size > len(self.points):
            raise Exception("The size of the batch exceeds the entire dataset. Reduce it!")

        prev_parameters = self.parameters.copy()
        prev_MSE = 99999999999
        for i in range(0, 100000):
            self.cnt_iterations += 1

            batch = self.points.sample(n=self.batch_size, replace=False)

            grad_const, grad_parameters = self._cnt_sum_of_loose_function(batch)

            for param in self.parameters.index:
                if param == "const":
                    self.parameters[param] -= self.step * grad_const
                else:
                    self.parameters[param] -= self.step * grad_parameters[param]

            new_norma = grad_parameters.dot(grad_parameters) ** 0.5
            if self.full_output:
                row = {
                    "Iteration": self.cnt_iterations,
                    "Step": self.step,
                    "norma": new_norma
                }

                row.update({
                    k: float(v)
                    for k, v in self.parameters.to_dict().items()
                })
                self.history.append(row)

            # stop
            if (self.parameters - prev_parameters).abs().sum() < epsilon:
                break

            if self._cnt_MSE() < 1e-9:
                break

            if abs(prev_MSE - self._cnt_MSE()) < epsilon:
                break

            prev_parameters = self.parameters.copy()
            prev_MSE = self._cnt_MSE()
            if self.choose_step_mode == "exponential_decay":
                self.step = self.start_step * np.exp((-self.exponential_decay) * self.cnt_iterations)
            elif self.choose_step_mode == "polynomial_decay":
                self.step = self.start_step / (
                        (self.polynomial_decay_beta * self.cnt_iterations + 1) ** self.polynomial_decay_alpha)

            if new_norma == history_last_norma:
                cnt_history_last_norma += 1
            else:
                cnt_history_last_norma = 0
                history_last_norma = new_norma

            if cnt_history_last_norma >= 10:
                cnt_history_last_norma = 0
                history_last_norma = -1
                print("Мы обнаружили зацикливание")
                if self.choose_step_mode == "constant":
                    self.print_history()

                    raise Exception(
                        f"A short circuit has been detected, and the constant step mode does not allow changing the step. Run the method again, but reduce the step. Current step: {self.step}")
                elif self.choose_step_mode == "piecewise_constant":
                    self.step /= 2
                elif self.choose_step_mode == "exponential_decay":
                    None
                else:
                    self.print_history()
                    raise Exception(
                        f"A short circuit has been detected")

        else:
            self.print_history()
            print(self.parameters)
            raise Exception(
                f"Protection has been activated. More than {self.cnt_iterations} iterations have been done.")

        self.print_history()
        end_time = time.time()
        self.print_results(start_time, end_time)
        return self.parameters

    def print_history(self):
        if self.full_output:
            df = pandas.DataFrame(self.history)
            display(df)

    def print_results(self, start_time, end_time):
        """
        Вывод результатов

        :param end_time: время конца запуска алгоритма
        :param start_time : время начала запуска алгоритма
        :return:
        """
        print(f"Расчёт параметра закончен. Был использован {self.choose_step_mode} шаг")
        print(f"Время выполнения: {end_time - start_time} секунд")
        print(f"С размером Батча: {self.batch_size} строк")
        print(f"Использованное количество итераций: {self.cnt_iterations}")
        for sym, value in self.parameters.items():
            try:
                print(f"{sym}: {round(value, 10)}")
            except ZeroDivisionError as ex:
                print(f"{sym}: {0}")
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        total_memory = sum(stat.size for stat in top_stats)
        print(f"Общая использованная память: {total_memory / 1024 / 1024:.2f} MB")

        print("[ Top 10 потребителей памяти ]")
        for stat in top_stats[:10]: # TODO на удаление
            print(stat)
        print("=================")

    def _cnt_sum_of_loose_function(self, batch: pandas.DataFrame) -> (pandas.Series, pandas.Series):
        all_mistakes = self._cnt_all_point_mistakes(batch)
        grad_parameters = (2 / (len(batch))) * batch.drop(columns='DEPENDENT').transpose().dot(all_mistakes)
        grad_const = (2 / (len(batch))) * all_mistakes.sum()
        return grad_const, grad_parameters

    def _cnt_all_point_mistakes(self, batch: pandas.DataFrame) -> pandas.Series:
        all_mistakes = []
        for _, point in batch.iterrows():
            y_pred = self.parameters["const"]
            y_real = point["DEPENDENT"]

            for name, value in point.items():
                if name != "DEPENDENT":
                    y_pred += value * self.parameters[name]

            mistake = y_pred - y_real

            all_mistakes.append(mistake)
        return pandas.Series(all_mistakes, index=batch.index)

    def _cnt_MSE(self) -> float:
        X = self.points.drop(columns="DEPENDENT")
        y = self.points["DEPENDENT"]
        pred = self.parameters["const"] + X.dot(self.parameters.drop("const"))
        return ((pred - y) ** 2).mean()