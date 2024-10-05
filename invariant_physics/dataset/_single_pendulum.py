
from .ode import *
from ._utils import params_random, params_default

PI = 3.14159_26535_89793_23846_26433_83279_50288_41971_69399_37510

PARAMS_CONFIG = {
    "task": "Single_Pendulum",
    "env_max": 5,
    "ode_dim": 2,
    "ode_dim_function": 2,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [1.00],
    "default_params_list": [
        [1.00],
        [0.90],
        [0.80],
        [1.10],
        [1.20],
    ],
    "random_y0_base": [0.25 * PI, 0.00],
    "default_y0_list": [
        [0.25 * PI, 0.00],
        [-0.25 * PI, 0.50],
        [0.125 * PI, 1.00],
        [-0.125 * PI, -0.50],
        [0.00 * PI, -1.00],
    ],
    "dt": 0.01,
    "t_min": 0,
    "t_max": 20,
    "curve_names": ["x", "y"],
    # "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->x', 'A->y', 'A->-10*sin(x)'],
    # "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)',
    #                 'A->x', 'A->y', 'A->sin(x)'],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)',
                 'A->x', 'A->y', 'A->sin(B)', 'A->cos(B)', 'A->log(B)', 'B->B*C', 'B->B+B', 'B->B-B', 'B->x', 'B->y'],
    "ntn_list": ['A', 'B'],
    "truth_ode_format": ["y", "-10 / {0} * sin(x)"],
    "purification_threshold": 0.05,
}


class SinglePendulum(ODEDataset):
    params_config = PARAMS_CONFIG
    
    def __init__(self, args):
        super(SinglePendulum, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        # alpha, beta, gamma, delta = iter(self.params[env_id])
        l = self.params[env_id][0]
        g = 10.00
        dy = np.asarray([
            x[1],
            -g / l * np.sin(x[0]),
        ])
        return dy


if __name__ == "__main__":
    pass
