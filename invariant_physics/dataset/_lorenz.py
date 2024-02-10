
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Lorenz",
    "env_max": 10,
    "ode_dim": 3,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [28.00, 10.00, 2.67],
    "default_params_list": [
        [28.00, 10.00, 2.67],
        [13.00, 9.90, 2.70],
        [15.00, 10.20, 2.74],
        [14.00, 10.50, 2.54],
        [19.00, 8.90, 3.00],
        [24.00, 9.70, 2.84],
        [18.00, 9.30, 2.60],
        [25.00, 9.80, 2.67],
        [17.00, 10.10, 2.76],
        [15.50, 10.40, 2.61],
    ],
    "random_y0_base": [6.00, 6.00, 15.00],
    "default_y0_list": [
        [6.00, 6.00, 15.00],
        [5.00, 7.00, 12.00],
        [5.80, 6.30, 17.00],
        [6.05, 6.40, 14.00],
        [6.25, 6.50, 11.00],
        [6.30, 6.10, 10.00],
        [6.20, 6.80, 18.00],
        [6.10, 6.90, 19.00],
        [5.90, 6.60, 20.00],
        [5.80, 5.80, 12.50],
    ],
    "dt": 0.001,
    "t_min": 0,
    "t_max": 10,
    "curve_names": ["x", "y", "z"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)', 
                'A->x', 'A->y','A->z'],
    "ntn_list": ['A'],
    "truth_ode_format": [
        "{1}*y-{1}*x",
        "{0}*x-x*z-y",
        "x*y-{2}*z",
    ]
}


class Lorenz(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(Lorenz, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        rho, sigma, beta = iter(self.params[env_id])
        dy = np.asarray([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])
        return dy


if __name__ == "__main__":
    pass
