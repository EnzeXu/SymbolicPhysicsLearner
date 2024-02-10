import numpy as np
import copy

from scipy.optimize import minimize

from ._utils import time_limit, sigmoid
from .production_rule_utils import simplify_eqs

from ..loss import VF_Loss

import torch

def combine_rewards_original(r_diff_list, r_parsimony_list, 
                            error_tolerance=None,
                            combine_operator="min"):
    # SPL, reward 0
    if combine_operator == "min":
        combine_operator = min
    elif combine_operator == "average" or combine_operator == "mean":
        combine_operator = np.mean
    return combine_operator(
        [rd*rp 
         for rd, rp in zip(r_diff_list, r_parsimony_list)]
    )

def combine_rewards_epsilon_piecewise(r_diff_list, r_parsimony_list, 
                                      error_tolerance=0.99,
                                      combine_operator="min"):
    # reward 1
    if combine_operator == "min":
        combine_operator = min
    elif combine_operator == "average" or combine_operator == "mean":
        combine_operator = np.mean
    return combine_operator(
        [0.5*rd if rd < error_tolerance else 0.5*rd + 0.5*rp
         for rd, rp in zip(r_diff_list, r_parsimony_list)]
    )

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def combine_rewards_epsilon_sigmoid(r_diff_list, r_parsimony_list, 
                                    error_tolerance=0.5,
                                    min_lam_diff=0.,
                                    combine_operator="min"):
    # reward 2
    if combine_operator == "min":
        combine_operator = min
    elif combine_operator == "average" or combine_operator == "mean":
        combine_operator = np.mean
    
    rs = []
    for rd, rp in zip(r_diff_list, r_parsimony_list):
        lam = (1-min_lam_diff) * sigmoid(3 / (min(error_tolerance, 1-error_tolerance)) * (rd-error_tolerance))
        rs.append(
            (1-lam)*rd + lam*rp
        )
    return combine_operator(rs)

def combine_rewards_epsilon_sigmoid_before(r_diff_list, r_parsimony_list, 
                                    error_tolerance=0.5,
                                    min_lam_diff=0.,
                                    combine_operator="min"):
    if combine_operator == "min":
        combine_operator = min
    elif combine_operator == "average" or combine_operator == "mean":
        combine_operator = np.mean
    
    rs = []
    for rd, rp in zip(r_diff_list, r_parsimony_list):
        lam = (1-min_lam_diff) * sigmoid(3 / (error_tolerance/2) * (rd-error_tolerance/2))
        rs.append(
            (1-lam)*rd + lam*rp
        )
    return combine_operator(rs)


def score_with_est(eq, tree_size, data_list, 
                   combine_rewards=combine_rewards_original,
                   task_ode_num=1, t_limit = 1.0, eta=0.999, 
                   reward_rescale=False, error_tolerance=0.5,
                   combine_operator="min", loss_func="L2", data_t_series=None):
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
        time limit (seconds) for single evaluation, default 1 second. 
        
    Returns
    -------
    score: Float
        discovered equations. 
    eq: Str
        discovered equations with estimated numerical values. 
    """
    # For VF
    assert loss_func in ["L2", "VF"]
    if loss_func == "VF":
        assert all(param is not None for param in [data_t_series])
    vf_criterion: VF_Loss = VF_Loss(func_num=100, integ_method='simps')
    # print(f"input data_list[0]: {data_list[0][:20]}")

    r_diff_list, r_parsimony_list, eqs = [], [], []
    if isinstance(eq, list):
        initial_eqs = eq
    else:
        initial_eqs = [eq] * len(data_list)
    for env, data in enumerate(data_list):
        # print(f"[normal] one data shape: {data.shape} ")
        eq = initial_eqs[env]

        ## define independent variables and dependent variable
        variables = list(data.columns)
        first_d_idx = variables.index('d'+variables[1])
        for variable in variables[:first_d_idx]:
            globals()[variable] = data[variable].to_numpy()
        target_variable = variables[first_d_idx + task_ode_num - 1]
        origin_variable = variables[1 + task_ode_num - 1]
        globals()['f_true'] = data[target_variable].to_numpy()
        globals()['y_true'] = data[origin_variable].to_numpy()
        # print(f"@@@@@ data[target_variable].to_numpy()={data[target_variable].to_numpy()[:20]}")

        ## count number of numerical values in eq
        c_count = eq.count('C')
        # start_time = time.time()
        with time_limit(t_limit, 'sleep'):
            try: 
                if c_count == 0:       ## no numerical values
                    f_pred = eval(eq)
                elif c_count >= 10:    ## discourage over complicated numerical estimations
                    return 0, eq
                else:                  ## with numerical values: coefficient estimationwith Powell method

                    # eq = prune_poly_c(eq)
                    c_lst = ['c'+str(i) for i in range(c_count)]
                    for c in c_lst: 
                        eq = eq.replace('C', c, 1)

                    def eq_test(c):
                        for i in range(len(c)): globals()['c'+str(i)] = c[i]
                        if loss_func == "VF":
                            xx = np.stack([y_true], axis=1)
                            # if len(f_pred.shape) == 2:
                            #     f_pred = f_pred[0]
                            ff = np.stack([eval(eq)], axis=1)
                            # print(f"x shape:{x.shape}, f shape:{f.shape}, f_pred shape {f_pred.shape} data_t_series shape: {data_t_series.shape}")
                            # print(f"f_pred data {f_pred[:5]}")
                            eq_diff = vf_criterion(ff, xx, data_t_series)
                            # print(eq_diff)
                        else:
                            eq_diff = np.linalg.norm(eval(eq) - f_true, 2)
                        return eq_diff
                    # def eq_test(c):
                    #     for i in range(len(c)): globals()['c'+str(i)] = c[i]
                    #     # print(f"loss_func: {loss_func}")
                    #     if loss_func == "VF":
                    #         # print(f"f_true shape: {f_true.shape}")
                    #         # print(f"eval(eq) shape: {eval(eq).shape}")
                    #         x = np.stack([f_true], axis=1)
                    #         f = eval(eq)
                    #         f = np.stack([f], axis=1)
                    #         # print("what?")
                    #         print(f.shape, x.shape, t_series.shape)
                    #         loss = criterion(f, x, t_series)
                    #         print(f"VF loss={loss}")
                    #         # print(f"eq='{eq}'")
                    #         # loss = loss_func(
                    #         #     ode_func=ode_func,
                    #         #     X=torch.tensor(data, dtype=torch.float32),
                    #         #     tspan=torch.tensor(t_series, dtype=torch.float32))
                    #     else:
                    #         loss = criterion(eval(eq) - f_true, 2)
                    #         # print(f"L2 loss={loss}")
                    #     return loss

                    x0 = [1.0] * len(c_lst)
                    c_lst = minimize(eq_test, x0, method='Powell', tol=1e-6).x.tolist() 
                    eq_est = eq
                    for i in range(len(c_lst)):
                        eq_est = eq_est.replace('c'+str(i), str(c_lst[i]), 1)
                    eq = eq_est.replace('+-', '-')
                    f_pred = eval(eq)
                    # print(f"[normal] output f_pred shape: {f_pred.shape} value: {f_pred[:10]}")
            except: 
                return 0, eq
        
        # if loss_func == "VF" and not isinstance(f_pred, float):
        #     # print(f"f_true shape:{f_true.shape}, f_pred shape:{f_pred.shape}, data_t_series shape: {data_t_series.shape}")
        #     if len(f_pred.shape) > 1:
        #         n_f_pred = f_pred.shape[-1]
        #         reward_diff_sum = 0
        #         x = np.stack([f_true], axis=1)
        #         for k in range(n_f_pred):
        #             f = np.stack([f_pred[:, k]], axis=1)
        #             reward_diff_sum += vf_criterion(x, f, data_t_series)
        #         r_diff = reward_diff_sum / n_f_pred
        #     else:
        #         x = np.stack([f_true], axis=1)
        #         f = np.stack([f_pred], axis=1)
        #         # print(f"x shape:{x.shape}, f shape:{f.shape}, data_t_series shape: {data_t_series.shape}")
        #         r_diff = vf_criterion(x, f, data_t_series)
        #     print(f"r_diff={r_diff} r_diff shape={r_diff.shape}")
        if loss_func == "VF":
            # print(f"[VF] f_pred shape: {f_pred.shape}")
            # import pdb;pdb.set_trace()
            # y_true_copy = copy.deepcopy(y_true)
            # f_pred_copy = copy.deepcopy(f_pred)
            xx = np.stack([y_true], axis=1)
            # if len(f_pred.shape) == 2:
            #     f_pred = f_pred[0]
            ff = np.stack([f_pred], axis=1)
            # print(f"x shape:{x.shape}, f shape:{f.shape}, f_pred shape {f_pred.shape} data_t_series shape: {data_t_series.shape}")
            # print(f"f_pred data {f_pred[:5]}")
            dis = vf_criterion(ff, xx, data_t_series)
            r_diff = float(1 / (1.0 + dis))
            assert isinstance(r_diff, float)
            # r_diff *= 1e-4
            # print(f"[VF] r_diff={r_diff}")
            # print(f"[VF] in VF, data_t_series_shape={data_t_series.shape} x.shape(truth)={x.shape} f.shape(from formula)={f.shape}")
            # draw_debug(x, f, data_t_series)
        else:  # L2
            # print(f"[L2] f_pred shape: {f_pred.shape} f_true shape: {f_true.shape} y_true shape: {y_true.shape}")
            diff = f_pred - f_true
            if reward_rescale:
                eps = 1e-6
                diff /= data[variables[task_ode_num]] + eps
            r_diff = float(1 / (1.0 + np.sqrt(np.linalg.norm(diff, 2) ** 2 / f_true.shape[0])))
            assert isinstance(r_diff, float)
            # print(f"[L2] r_diff={r_diff}")
        # print()
        r_parsimony = eta ** tree_size
        r_diff_list.append(r_diff)
        r_parsimony_list.append(r_parsimony)
        eqs.append(eq)   
#     print(r_diff_list, r_parsimony_list)
    try:
        return (
            combine_rewards(r_diff_list, r_parsimony_list,
                            error_tolerance=error_tolerance,
                            combine_operator=combine_operator),
            simplify_eqs(eqs)
        )
    except:
        print(eqs)
        raise


def draw_debug(x, f, ts):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 3))
    # print(f"ts={ts}")
    plt.plot(ts, x.flatten(), label="x (truth)")
    plt.plot(ts, f.flatten(), label="f (from formula)")
    plt.legend()
    file_path = "test/train_last.png" if len(ts) == 400 else "test/test_last.png"
    plt.savefig(file_path)