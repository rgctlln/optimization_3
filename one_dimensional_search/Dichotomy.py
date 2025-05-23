def dichotomy(func, a: float, b: float, epsilon: float = 1e-5, delta: float = 1e-6):
    """
    Метод дихотомии для поиска минимума функции одной переменной.

    :param func: Функция одной переменной
    :param a: Левая граница отрезка
    :param b: Правая граница отрезка
    :param epsilon: Точность остановки
    :param delta: Смещение от середины
    :return: Приближённое значение точки минимума
    """
    while (b - a) / 2.0 > epsilon:
        mid = (a + b) / 2.0
        x1 = mid - delta
        x2 = mid + delta
        f1 = func(x1)
        f2 = func(x2)

        if f1 < f2:
            b = x2
        else:
            a = x1

    return (a + b) / 2.0
