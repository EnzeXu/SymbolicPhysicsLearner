
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "SIRCovid",
    "env_max": 10,
    "ode_dim": 3,
    "ode_dim_function": 3,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [1.0, 1.0],
    "default_params_list": [
        [1.0, 1.0],
    ],
    "random_y0_base": [0.0, 0.0, 0.0],
    "default_y0_list": [
        [1.0, 1.0],
    ],
    "dt": 1,
    "t_min": 0,
    "t_max": 299,
    "curve_names": ["x", "y", "z"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A*C)',
                'A->x', 'A->y','A->z', 'A->C'],
    "ntn_list": ['A'],
    "truth_ode_format": [
        "-{0}*x*y",
        "{0}*x*y-{1}*y",
        "{1}*y",
    ],
    "purification_threshold": 0.10,
}


class SIRCovid(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(SIRCovid, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        return NotImplementedError


if __name__ == "__main__":
    pass
