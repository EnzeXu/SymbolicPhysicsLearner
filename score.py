import numpy as np
from numpy import *
from sympy import simplify, expand
from scipy.optimize import minimize
import threading
import _thread
import time

import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def simplify_eq(eq):
    return str(expand(simplify(eq)))


def prune_poly_c(eq):
    '''
    if polynomial of C appear in eq, reduce to C for computational efficiency.
    '''
    eq = simplify_eq(eq)
    if 'C**' in eq:
        c_poly = ['C**' + str(i) for i in range(10)]
        for c in c_poly:
            if c in eq: eq = eq.replace(c, 'C')
    return simplify_eq(eq)


def score_with_est(eq, tree_size, data, t_limit=1.0, eta=0.999, task_ode_num=0):
    """
    Calculate reward score for a complete parse tree
    If placeholder C is in the equation, also excute estimation for C
    Reward = 1 / (1 + MSE) * Penalty ** num_term

    Parameters
    ----------
    eq : Str object.
        the discovered equation (with placeholders for coefficients).
    tree_size : Int object.
        number of production rules in the complete parse tree.
    data : 2-d numpy array.
        measurement data, including independent and dependent variables (last row).
    t_limit : Float object.
        time limit (seconds) for ssingle evaluation, default 1 second.

    Returns
    -------
    score: Float
        discovered equations.
    eq: Str
        discovered equations with estimated numerical values.
    """
    assert task_ode_num >= 1
    variables = list(data.columns)
    first_d_idx = variables.index('d' + variables[1])
    for variable in variables[:first_d_idx]:
        globals()[variable] = data[variable].to_numpy()
    target_variable = variables[first_d_idx + task_ode_num - 1]
    origin_variable = variables[1 + task_ode_num - 1]
    globals()['f_true'] = data[target_variable].to_numpy()
    globals()['y_true'] = data[origin_variable].to_numpy()

    ## define independent variables and dependent variable
    # num_var = data.shape[0] - 1
    # if num_var <= 3:  ## most cases ([x], [x,y], or [x,y,z])
    #     current_var = 'x'
    #     for i in range(num_var):
    #         globals()[current_var] = data[i, :]
    #         current_var = chr(ord(current_var) + 1)
    #     globals()['f_true'] = data[-1, :]
    # else:  ## currently only double pendulum case has more than 3 independent variables
    #     globals()['x1'] = data[0, :]
    #     globals()['x2'] = data[1, :]
    #     globals()['w1'] = data[2, :]
    #     globals()['w2'] = data[3, :]
    #     globals()['wdot'] = data[4, :]
    #     globals()['f_true'] = data[5, :]

    ## count number of numerical values in eq
    c_count = eq.count('C')
    # start_time = time.time()
    try:
        with time_limit(t_limit):
            if c_count == 0:  ## no numerical values
                f_pred = eval(eq)
            elif c_count >= 4:  ## discourage over complicated numerical estimations
                return 0, eq
            else:  ## with numerical values: coefficient estimationwith Powell method
                # eq = prune_poly_c(eq)
                c_lst = ['c' + str(i) for i in range(c_count)]
                for c in c_lst:
                    eq = eq.replace('C', c, 1)

                def eq_test(c):
                    for i in range(len(c)): globals()['c' + str(i)] = c[i]
                    return np.linalg.norm(eval(eq) - f_true, 2)

                x0 = [1.0] * len(c_lst)
                c_lst = minimize(eq_test, x0, method='Powell', tol=1e-6).x.tolist()
                c_lst = [np.round(x, 10) if abs(x) > 1e-2 else 0 for x in c_lst]
                eq_est = eq
                for i in range(len(c_lst)):
                    eq_est = eq_est.replace('c' + str(i), str(c_lst[i]), 1)
                eq = eq_est.replace('+-', '-')
                f_pred = eval(eq)
        # print("Not TLE")
    except:
        # print(f"TLE for {t_limit}s")
        return 0, eq
    # print(f"f_pred shape: {f_pred.shape}")
    r = float(eta ** tree_size / (1.0 + np.linalg.norm(f_pred - f_true, 2) ** 2 / f_true.shape[0]))

    # run_time = np.round(time.time() - start_time, 3)
    # print('runtime :', run_time,  eq,  np.round(r, 3))

    return r, eq