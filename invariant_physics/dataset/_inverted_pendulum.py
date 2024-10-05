
from .ode import *
from ._utils import params_random, params_default

# PI = 3.14159_26535_89793_23846_26433_83279_50288_41971_69399_37510

PARAMS_CONFIG = {
    "task": "Inverted_Pendulum",
    "env_max": 5,
    "ode_dim": 5,
    "ode_dim_function": 5,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [-0.18181818181818182, 2.672727272727274, -0.45454545454545464, 31.181818181818187],
    "default_params_list": [
        [-0.18181818181818182, 2.672727272727274, -0.45454545454545464, 31.181818181818187],
        [-0.38095238095238093, 1.4000000000000001, -1.4285714285714284, 41.99999999999999],
        [-0.3076923076923077, 2.261538461538462, -0.576923076923077, 22.615384615384617],
        [-0.4444444444444445, 3.2666666666666666, -0.833333333333333, 24.49999999999999],
        [-0.12903225806451615, 2.8451612903225807, -0.48387096774193544, 47.41935483870967],
    ],
    "random_y0_base": [0.00, 0.0, 0.05, 0.00],
    "default_y0_list": [
        [0.00, 0.0, 0.05, 0.00],
    ],
    "dt": 0.01,
    "t_min": 0,
    "t_max": 20,
    "curve_names": ["x", "y", "z", "w", "a"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)',
                 'A->x', 'A->y', 'A->z', 'A->w', 'A->1.0'],
    "ntn_list": ['A', 'B'],
    "truth_ode_format": ["y", "{0} * y + {1} * z", "w", "{2} * y + {3} * z"],
    "purification_threshold": 0.05,
}


class InvertedPendulum(ODEDataset):
    params_config = PARAMS_CONFIG
    
    def __init__(self, args):
        super(InvertedPendulum, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        # alpha, beta, gamma, delta = iter(self.params[env_id])
        c1, c2, c3, c4 = iter(self.params[env_id])
        dy = np.asarray([
            x[1],
            c1 * x[1] + c2 * x[2],
            x[3],
            c3 * x[1] + c4 * x[2],
        ])
        return dy


if __name__ == "__main__":
    pass
