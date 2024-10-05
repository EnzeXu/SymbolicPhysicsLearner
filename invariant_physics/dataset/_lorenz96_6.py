
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Lorenz96_6",
    "env_max": 10,
    "ode_dim": 6,
    "ode_dim_function": 6,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [8.0],
    "default_params_list": [
        [8.0],
        [9.0],
        [10.0],
        [11.0],
        [12.0],
        [13.0],
        [14.0],
        [15.0],
        [16.0],
        [17.0],
    ],
    "random_y0_base": [8.1, 8.00, 8.00, 8.00, 8.00, 8.00],
    "default_y0_list": [
        [8.1, 8.00, 8.00, 8.00, 8.00, 8.00],
        [9.1, 9.00, 9.00, 9.00, 9.00, 9.00],
        [10.1, 10.00, 10.00, 10.00, 10.00, 10.00],
        [11.1, 11.00, 11.00, 11.00, 11.00, 11.00],
        [12.1, 12.00, 12.00, 12.00, 12.00, 12.00],
        [13.1, 13.00, 13.00, 13.00, 13.00, 13.00],
        [14.1, 14.00, 14.00, 14.00, 14.00, 14.00],
        [15.1, 15.00, 15.00, 15.00, 15.00, 15.00],
        [16.1, 16.00, 16.00, 16.00, 16.00, 16.00],
        [17.1, 17.00, 17.00, 17.00, 17.00, 17.00],
    ],
    "dt": 0.001,
    "t_min": 0,
    "t_max": 10,
    "curve_names": ["a", "b", "c", "d", "e", "f"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)', 
                'A->C', 'A->a', 'A->b', 'A->c', 'A->d', 'A->e', 'A->f'],
    "ntn_list": ['A'],
    "truth_ode_format": [
        "1.0*b*f-1.0*e*f-1.0*a+{0}",
        "1.0*c*a-1.0*f*a-1.0*b+{0}",
        "1.0*d*b-1.0*a*b-1.0*c+{0}",
        "1.0*e*c-1.0*b*c-1.0*d+{0}",
        "1.0*f*d-1.0*c*d-1.0*e+{0}",
        "1.0*a*e-1.0*d*e-1.0*f+{0}",
    ],
    "purification_threshold": 0.10,
}


class Lorenz96_6(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(Lorenz96_6, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        F = self.params[env_id][0]
        dy = np.asarray([
            (x[1] - x[4]) * x[5] - x[0] + F,
            (x[2] - x[5]) * x[0] - x[1] + F,
            (x[3] - x[0]) * x[1] - x[2] + F,
            (x[4] - x[1]) * x[2] - x[3] + F,
            (x[5] - x[2]) * x[3] - x[4] + F,
            (x[0] - x[3]) * x[4] - x[5] + F,
        ])
        return dy


if __name__ == "__main__":
    pass
