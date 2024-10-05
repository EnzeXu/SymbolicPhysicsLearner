
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Acceleration_Fixed_Mass",
    "env_max": 5,
    "ode_dim": 2,
    "ode_dim_function": 1,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [0.20],
    "default_params_list": [
        [0.30],
        [0.25],
        [0.20],
        [0.15],
        [0.10],
    ],
    "random_y0_base": [0.00],
    "default_y0_list": [
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
    ],
    "dt": 0.001,
    "t_min": 0,
    "t_max": 10,
    "curve_names": ["x", "y"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)', 
                'A->x', 'A->y', 'A->z', 'A->C'],
    "ntn_list": ['A'],
    "truth_ode_format": ["x-{0}*y-{0}*9.80"],
    "purification_threshold": 0.03,
}


class AccelerationFixedMass(ODEDataset):
    params_config = PARAMS_CONFIG
    
    def __init__(self, args):
        super(AccelerationFixedMass, self).__init__(args, PARAMS_CONFIG, non_ode_function=True)

    def _func(self, x, t, env_id):
        mu = self.params[env_id][0]
        F1 = x[:, 0: 1]
        F2 = x[:, 1: 2]
        # m = x[:, 2: 3]
        y = (F1 - F2 * mu) / 1.0 - 9.80 * mu
        return y

    # def _set_non_ode_y(self):
    #     self.


if __name__ == "__main__":
    pass
