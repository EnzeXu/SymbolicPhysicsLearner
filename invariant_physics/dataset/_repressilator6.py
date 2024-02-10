
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Repressilator6",
    "env_max": 10,
    "ode_dim": 6,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [10.00, 3.00, 0.0000010, 1.00],
    "default_params_list": [
        [10.00, 3.00, 0.0000010, 1.00],
        [9.90, 2.90, 0.0000011, 0.95],
        [10.10, 3.10, 0.0000012, 0.90],
        [10.30, 3.20, 0.0000013, 0.85],
        [10.20, 3.30, 0.0000014, 1.05],
        [9.80, 2.80, 0.0000015, 1.10],
        [9.70, 2.70, 0.0000009, 1.15],
        [9.60, 3.40, 0.0000008, 1.20],
        [10.40, 3.50, 0.0000007, 1.25],
        [10.50, 2.60, 0.0000006, 0.85],
    ],
    "random_y0_base": [9.10, 0.90, 0.16, 7.24, 2.85, 0.16],
    "default_y0_list": [
        [9.10, 0.90, 0.16, 7.24, 2.85, 0.16],
        [9.12, 0.80, 0.15, 7.14, 2.75, 0.14],
        [9.33, 0.70, 0.14, 7.00, 2.60, 0.12],
        [9.54, 0.60, 0.13, 6.90, 2.55, 0.10],
        [9.85, 0.50, 0.12, 6.85, 2.40, 0.08],
        [9.01, 0.95, 0.11, 7.05, 2.35, 0.18],
        [8.76, 1.00, 0.17, 7.80, 2.15, 0.20],
        [8.50, 1.05, 0.18, 7.70, 2.00, 0.22],
        [8.40, 1.10, 0.19, 7.60, 3.00, 0.24],
        [9.90, 1.15, 0.20, 7.50, 3.25, 0.26],
    ],
    "dt": 0.005,
    "t_min": 0,
    "t_max": 20,
    "curve_names": ["x", "y", "z", "u", "v", "w"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->A**C',
                'A->x', 'A->y','A->z','A->C','A->u','A->v','A->w'],
    "ntn_list": ['A'],
    "truth_ode_format": [
        "{0}/({2}+1/(1+w**{1}))-x",
        "{0}/({2}+1/(1+u**{1}))-y",
        "{0}/({2}+1/(1+v**{1}))-z",
        "{3}*x-{3}*u",
        "{3}*y-{3}*v",
        "{3}*z-{3}*w",
    ],
}


class Repressilator6(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(Repressilator6, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        beta, n, rho, gamma = iter(self.params[env_id])
        dy = np.asarray([
            beta * (rho + 1 / (1 + x[5] ** n)) - x[0],
            beta * (rho + 1 / (1 + x[3] ** n)) - x[1],
            beta * (rho + 1 / (1 + x[4] ** n)) - x[2],
            gamma * (x[0] - x[3]),
            gamma * (x[1] - x[4]),
            gamma * (x[2] - x[5])
        ])
        return dy


if __name__ == "__main__":
    pass
