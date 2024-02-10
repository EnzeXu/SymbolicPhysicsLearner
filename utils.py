import datetime
import pytz
import sympy as sp
import random
import numpy as np
from sympy import symbols, sympify, Mul


def get_now_string(time_string="%Y%m%d_%H%M%S_%f"):
    # return datetime.datetime.now().strftime(time_string)
    est = pytz.timezone('America/New_York')

    # Get the current time in UTC and convert it to EST
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    est_now = utc_now.astimezone(est)

    # Return the time in the desired format
    return est_now.strftime(time_string)


def is_term_constant(term):
    return isinstance(term, (sp.Float, sp.Integer, sp.Rational))


def extract(expression):
    """
    Extract the `full_terms`, `terms`, and `coefficient_terms` (all in list) of an expression
    The order of each part is by str comparison of items in `terms`
    :param expression: e.g., "3*y+2*sin(x)-3*x**2+2*x*y"
    :return: [2*sin(x), -3*x**2, 2*x*y, 3*y], [sin(x), x**2, x*y, y], [2, -3, 2, 3]
    """
    expr = sp.sympify(expression)
    raw_terms = list(sp.Add.make_args(expr))
    # print(raw_terms)
    result = []

    for term in raw_terms:
        term = sp.sympify(term)
        # if is_term_constant(term):
        #     continue
        # print(sp.Mul.make_args(term))
        coefficient_list = []
        non_coefficient_list = []
        for factor in sp.Mul.make_args(term):
            if is_term_constant(factor):
                coefficient_list.append(factor)
            else:
                non_coefficient_list.append(factor)
            # print(f"{factor}: {is_term_constant(factor)}")
        coefficient_part = sp.sympify(sp.Mul(*[sp.sympify(item) for item in coefficient_list]))
        non_coefficient_part = sp.sympify(sp.Mul(*[sp.sympify(item) for item in non_coefficient_list]))
        result.append([term, non_coefficient_part, coefficient_part])

    result = sorted(result, key=lambda x: str(x[1]))
    full_terms = [item[0] for item in result]
    terms = [item[1] for item in result]
    coefficient_terms = [item[2] for item in result]
    return full_terms, terms, coefficient_terms


def setup_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def remove_constant(one_list):
    return [item for item in one_list if item != "1" and item != 1]