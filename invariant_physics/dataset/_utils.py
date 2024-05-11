import random
import os
import sympy as sp
import datetime
import numpy as np
import pandas as pd
import pytz
import re
from scipy.stats import qmc
from sympy import symbols, sympify, Mul
from scipy.integrate import odeint

def sample_lhs(lb, ub, n, skip=1):
    large_n = skip * n
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=large_n)
    sample = qmc.scale(sample, lb, ub)
    sample = np.sort(np.squeeze(sample))
    sample = sample[::skip]
    return sample


def params_default(num_env, **kwargs):
    default_list = kwargs["default_list"]
    return default_list[:num_env]


def params_random(num_env, **kwargs):
    base, random_rate, seed = kwargs["base"], kwargs["random_rate"], kwargs["seed"]
    params = []
    for idx in range(num_env):
        random.seed(seed + idx)
        params.append([round(random.uniform((1 - random_rate) * item, (1 + random_rate) * item), 7) for item in base])
    return params



def save_to_csv(save_path, cols, headers=None):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if headers:
        assert len(cols) == len(headers)
        df = pd.DataFrame({f"{headers[i]}": cols[i] for i in range(len(cols))})
        df.to_csv(save_path, index=False)
    else:
        df = pd.DataFrame({f"item {i}": cols[i] for i in range(len(cols))})
        df.to_csv(save_path, index=False, header=False)


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


def score_match_terms(term_list_1: list, term_list_2: list):
    term_list_1 = list(set(term_list_1))
    term_list_2 = list(set(term_list_2))
    match_cnt = sum([int(item in term_list_2) for item in term_list_1])
    return match_cnt / max(len(term_list_1), len(term_list_2))


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


def evaluate_expression(expression_str, variable_list, variable_values):
    # print("debug:", expression_str, variable_list, variable_values)
    variables = symbols(variable_list)
    expr = sympify(expression_str)
    variable_dict = {var: val for var, val in zip(variables, variable_values)}
    result = expr.subs(variable_dict)
    result_value = result.evalf()
    return result_value


def remove_constant(one_list):
    return [item for item in one_list if item != "1"]


def determine_most_frequent_terms(input_dic: dict):
    assert len(input_dic) > 0
    dic = dict()
    vals = input_dic.values()
    for item in vals:
        sub_vals = item.values()
        for sub_item in sub_vals:
            if sub_item["purified_predicted_terms"] not in dic:
                dic[sub_item["purified_predicted_terms"]] = [sub_item["purified_predicted_terms"], 1]
            else:
                dic[sub_item["purified_predicted_terms"]][1] += 1
    sorted_values = sorted(list(dic.values()), key=lambda x: -x[1])
    print(sorted_values)
    return sorted_values[0][0]


def transform_sympy(input_eq_str):
    expr = sympify(input_eq_str)
    C = symbols('C')
    transformed_terms = []
    for term in expr.as_ordered_terms():
        coeff, rest = term.as_coeff_Mul()
        if rest == 1:
            transformed_terms.append(C)
        else:
            transformed_terms.append(Mul(C, rest))
    transformed_expr = sum(transformed_terms)
    return str(transformed_expr)

def set_eq_precision(eq_str, digit):
    expr = sympify(eq_str)
    C = symbols('C')
    transformed_terms = []
    for term in expr.as_ordered_terms():
        coeff, rest = term.as_coeff_Mul()
        rounded_coeff = round(coeff, digit)
        if rest == 1:
            transformed_terms.append(rounded_coeff)
        else:
            transformed_terms.append(rounded_coeff * rest)
    transformed_expr = sum(transformed_terms)
    return str(transformed_expr)


def evaluate_eq_into_value(eq_str, curve_names, data_points):
    results = np.zeros(data_points.shape[0])
    for i, point in enumerate(data_points):
        var_values = {var_name: value for var_name, value in zip(curve_names, point)}
        result = eval(eq_str, {}, var_values)
        results[i] = result
    return results


def evaluate_trajectory_rmse(ode, eq_str, i_env, task_ode_num):
    # data_points = ode.y_noise[i_env][ode.test_indices_list[i_env]]
    data_points = ode.y_noise[i_env]
    dy_prediction = evaluate_eq_into_value(eq_str, ode.params_config["curve_names"], data_points)
    # dy_truth = ode.dy_noise[i_env][ode.test_indices_list[i_env], task_ode_num - 1]
    dy_truth = ode.dy_noise[i_env][:, task_ode_num - 1]
    # print(f"dy_prediction shape: {dy_prediction.shape} dy_truth shape: {dy_truth.shape}")
    assert dy_prediction.shape == dy_truth.shape
    mse = np.mean((dy_prediction - dy_truth) ** 2)
    rmse = np.sqrt(mse)
    variance = np.var(dy_truth)
    std_deviation = np.sqrt(variance)
    relative_mse = mse / variance if variance != 0 else float('inf')
    relative_rmse = rmse / std_deviation if std_deviation != 0 else float('inf')
    return mse, rmse, relative_mse, relative_rmse


def get_train_test_total_list(train_test_total: str, num_env: int, seed=None):
    """
    Args:
        train_test_total: supports three types of argument:
            (1) one integer, like 500, indicating it's a balanced dataset;
            (2) integers split by "/", like 500/400/300/200/100. There would be an error if its length mismatches the num_env;
            (3) a string "default_x", following a built-in dict as below.
        num_env: an integer, number of environment
        seed: an integer for generating random ordering list
    Returns:
        a list of environment size, e.g., [500, 400, 300, 200, 100].
    """
    default_dic = {
        "default_0": "500/500/500/500/500",
        "default_1": "20/20/20/20/20",
        "default_2": "100/100/100/100/100",
        "default_3": "500/400/40/20/10",
        "default_4": "500/80/40/20/10",
        "default_5": "100/50/50/20/10",
        "default_6": "500/400/300/200/100",
        "default_7": "500/400/300/200/10",
        "default_8": "500/400/300/20/10",
        "default_10": "200/200/200/200/200",
        "default_11": "500/400/40/40/20",
        "default_12": "500/200/200/50/50",
        "default_13": "500/125/125/125/125",
    }

    if train_test_total.isdigit():
        train_test_total_list = [int(train_test_total) for i in range(num_env)]
    else:
        if "/" not in train_test_total:
            assert train_test_total in default_dic, "Error: key error in get_train_test_total_list: " + str(train_test_total)
            string = default_dic[train_test_total]
        else:
            string = train_test_total
        parts = string.split("/")
        train_test_total_list = [int(item) for item in parts]
    assert len(train_test_total_list) == num_env, "Error: mismatching between " + str(train_test_total) + " and " + str(num_env)
    # print(train_test_total_list)
    one_order = generate_random_order(num_env, seed)
    train_test_total_list_swapped = reseat(train_test_total_list, one_order)
    print(f"Swap dataset size: {train_test_total_list} -> {train_test_total_list_swapped}")
    return train_test_total_list_swapped


def generate_random_order(n, seed=None):
    if seed is not None:
        random.seed(seed)
    numbers = list(range(n))
    random.shuffle(numbers)
    return numbers


def reseat(one_list, one_order):
    assert len(one_list) == len(one_order)
    return [one_list[one_order[i]] for i in range(len(one_order))]


if __name__ == "__main__":
    # get_train_test_total_list("default_1", 5)
    numbers = [10,20,30,0,50]
    order = generate_random_order(5, 500)
    print(order)
    print(reseat(numbers, order))
    # a = " -0.389927009553766*x*y + 1.1997924353311362*x"
    # a = "-1/(5*x + 1) * (1+2*x) - 1*x*(1-y)"
    # print(transform_sympy(a))
    # a = {
    #     "1": {
    #         "purified_predicted_terms": "aaaa"
    #     },
    #     "2": {
    #         "purified_predicted_terms": "bbb"
    #     },
    #     "3": {
    #         "purified_predicted_terms": "aaaa"
    #     },
    #     "4": {
    #         "purified_predicted_terms": "c"
    #     },
    # }
    # print(determine_most_frequent_terms(a))

    # expression = "x+y+z"
    # variables = ["x", "y", "z"]
    # values = [1.1, 2.0, -1.4]
    # print(evaluate_expression(expression, variables, values))

    # expr1 = "3*y+2*sin(x)-3*x**2+2.1*x*y"
    # expr2 = "15*x**2*y/(x + 1) - 13*x*2.0*y/(x + 25) + 5.1*x/(x + 5.0)"
    # parts = extract(expr1)
    # # print(parts)
    # print(score_match_terms(parts[1], parts[1]))
