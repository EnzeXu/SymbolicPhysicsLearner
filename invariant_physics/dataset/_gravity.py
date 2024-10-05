
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Gravity",
    "env_max": 10,
    "ode_dim": 1,
    "ode_dim_function": 1,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [9.80],
    "default_params_list": [
        [9.81],
        [1.62],
        [24.79],
        [3.71],
        [1.35],
        [0.78],
        [11.15],
        [10.44],
        [8.68],
        [0.62],
    ],
    "random_y0_base": [0.00],
    "default_y0_list": [
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
        [0.00],
    ],
    "dt": 0.01,
    "t_min": 0,
    "t_max": 10,
    "curve_names": ["x"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)', 
                'A->x', 'A->y'],
    "ntn_list": ['A'],
    "truth_ode_format": ["{0}*x"],
    "purification_threshold": 0.05,
}


class Gravity(ODEDataset):
    params_config = PARAMS_CONFIG
    
    def __init__(self, args):
        super(Gravity, self).__init__(args, PARAMS_CONFIG, non_ode_function=True)

    def _func(self, x, t, env_id):
        m = self.params[env_id][0]
        # dy = np.asarray([
        #     alpha * x[0] - beta * x[0] * x[1],
        #     delta * x[0] * x[1] - gamma * x[1],
        # ])
        y = m * x
        return y


if __name__ == "__main__":
    pass
