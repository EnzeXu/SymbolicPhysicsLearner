import signal
from contextlib import contextmanager
import numpy as np
import sympy as sp

from ..dataset import extract, evaluate_expression

class TimeOutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    def _handler(signum, frame):
        raise TimeOutException("Timed out for operation {}".format(msg))
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    except TimeOutException:
        raise 
    finally:
        signal.alarm(0)


def purify_strategy1(eq, data, variable_list, threshold=0.05):
    # data is in shape (N, m). Here m is the dimension of the ODE system
    # print(f"input: {eq}")
    full_terms, terms, _ = extract(eq)
    # print(full_terms)
    # print(terms)
    n = data.shape[0]
    abs_value_array = np.zeros([n, len(full_terms)])
    abs_ratio_array = np.zeros([n, len(full_terms)])
    for i in range(n):
        for j, one_full_term in enumerate(full_terms):
            abs_value_array[i][j] = np.abs(evaluate_expression(one_full_term, variable_list, data[i]))
        for j in range(len(full_terms)):
            abs_ratio_array[i][j] = abs_value_array[i][j] / np.sum(abs_value_array[i])
    avg_ratio = np.average(abs_ratio_array, axis=0)
    purified_full_terms = [full_terms[i] for i in range(len(full_terms)) if avg_ratio[i] >= threshold]
    print(f"full terms: {full_terms} ratio: {avg_ratio} threshold: {threshold} purified_full_terms: {purified_full_terms}")
    purified_eq = sp.sympify(sp.Add(*purified_full_terms))
    # print(avg_ratio)
    # print(f"output: {purified_eq}")
    return purified_eq

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    expression = "x+y+z"
    variables = ["x", "y", "z"]
    values = [1.1, 2.0, -1.4]
    print(evaluate_expression(expression, variables, values))
