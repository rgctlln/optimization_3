# from sympy import symbols, diff, lambdify
#
# x, y = symbols('x y')
#
# func = 3 * x ** 2 + y ** 2 - x * y - 4 * x
#
# print(type(func))
# # вычисляем градиент
# grad_x = diff(func, x)
# grad_y = diff(func, y)
#
# func_grad = lambdify((x, y), [grad_x, grad_y])
#
# # градиент
# print(f"Градиент (аналитически): [{grad_x}, {grad_y}]")
#
# # градиент в точке
# parameters = (10, 20)
# print(f"Градиент в точке {parameters}: {func_grad(*parameters)}")

import numpy as np
from scipy.optimize import minimize

# Определяем функцию для минимизации
def func_to_minimize(vars):
    x, y = vars
    return x ** 2 + y ** 2 - 1

# Начальные значения для x и y
initial_guess = [0, 0]

# Минимизация функции
result = minimize(func_to_minimize, initial_guess)

# Выводим результаты
print("Решение:", result.x)  # Координаты минимума
print("Минимальное значение функции:", result.fun)  # Значение функции в минимуме

