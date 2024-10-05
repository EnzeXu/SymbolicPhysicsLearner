
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Repressilator3",
    "env_max": 10,
    "ode_dim": 3,
    "ode_dim_function": 3,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [10.00, 3.00],
    "default_params_list": [
        [10.00, 3.00],
        [9.90, 2.00],
        [10.10, 3.00],
        [10.30, 3.00],
        [10.20, 3.00],
        [9.80, 2.00],
        [9.70, 2.00],
        [9.60, 3.00],
        [10.40, 3.00],
        [10.50, 2.00],
    ],
    "random_y0_base": [1.75, 4.77, 1.05],
    "default_y0_list": [
        [1.75, 4.77, 1.05],
        [2.00, 4.60, 1.15],
        [2.30, 4.70, 1.25],
        [2.15, 4.50, 1.02],
        [2.10, 4.40, 1.24],
        [2.05, 4.80, 1.01],
        [1.45, 4.10, 0.95],
        [1.55, 4.00, 0.94],
        [1.60, 5.20, 0.91],
        [1.70, 5.40, 0.90],
    ],
    "dt": 0.001,
    "t_min": 0,
    "t_max": 5,
    "curve_names": ["x", "y", "z"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->A**C',
                'A->x', 'A->y','A->z','A->C'],
    "ntn_list": ['A'],
    "truth_ode_format": ["{0}/(1+z**{1})-x", "{0}/(1+x**{1})-y", "{0}/(1+y**{1})-z"],
    "purification_threshold": 0.05,
}


class Repressilator3(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(Repressilator3, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        beta, n = iter(self.params[env_id])
        dy = np.asarray([
            beta / (1 + x[2] ** n) - x[0],
            beta / (1 + x[0] ** n) - x[1],
            beta / (1 + x[1] ** n) - x[2]
        ])
        return dy


if __name__ == "__main__":
    pass
