import numpy as np
import sympy
from sympy import diff, lambdify


class Function:
    """
    Ex:

    from sympy import symbols

    x, y = symbols('x y')

    func = Function(x**2 + y**2 - 1)

    print(func.evaluate(x=2, y=4))

    """

    def __init__(self, func: sympy.core):
        self.function = func
        self.list_var: tuple = tuple(func.free_symbols)
        temp_list = []
        for el in self.list_var:
            temp_list.append(el.name)
        self.list_str_var: tuple = tuple(sorted(temp_list))

        # Компиляция функции и градиентов
        self.f_np = lambdify(self.list_var, self.function, modules='numpy')
        self.grad_exprs = [diff(self.function, var) for var in self.list_var]
        self.grad_np = lambdify(self.list_var, self.grad_exprs, modules='numpy')

    def evaluate(self, vector_of_variables: dict) -> int:

        """
        Функция для подстановки чисел в выражение.

        Например, до этого передали выражение x^2+y^2, тогда вызов будет
            func.evaluate({x:4, y:5})

        :param vector_of_variables: аргументы, передаваемые в выражение как значения для символов;
        :return:
        """

        return self.function.subs(vector_of_variables)

    def _count_partial_derivative(self, partial_symbol: sympy.core.symbol.Symbol) -> sympy.core:
        """
        Функция для вычисления частной производной по заданному символу

        :param partial_symbol: символ, по которому происходит нахождение производной
        :return:
        """

        return diff(self.function, partial_symbol)

    def count_all_partial_derivative(self) -> dict:
        derivative = {}
        for el in self.list_var:
            derivative[el] = (self._count_partial_derivative(el))
        return derivative

    def get_tuple_of_variables(self) -> tuple:
        return self.list_var

    def grad(self, x: list | np.ndarray) -> np.ndarray:
        grad_values = self.grad_np(*x)
        return np.array(grad_values, dtype=np.float64)

    def __str__(self):
        return str(self.function).replace("**", "^")

    def __call__(self, x: list | np.ndarray) -> float:
        return float(self.f_np(*x))
