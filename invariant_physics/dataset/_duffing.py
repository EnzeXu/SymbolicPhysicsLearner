
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Duffing",
    "env_max": 5,
    "ode_dim": 3,
    "ode_dim_function": 3,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [0.2, -1.0, 1.0, 0.3, 1.2],
    "default_params_list": [
        [0.2, -1.0, 1.0, 0.3, 1.2],
        [0.15, -1.1, 0.4, 0.4, 1.0],
        [0.22, -1.2, 0.5, 0.2, 1.4],
        [0.17, -0.9, 0.6, 0.5, 1.3],
        [0.19, -0.8, 0.8, 0.6, 1.1],
    ],
    "random_y0_base": [1.0, 0.0, 0.0],
    "default_y0_list": [
        [1.0, 0.0],
    ],
    "dt": 0.001,
    "t_min": 0,
    "t_max": 100,
    "curve_names": ["x", "y", "z"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)',
                 'A->x', 'A->y', 'A->z', 'A->sin(B)', 'A->cos(B)', 'A->log(B)', 'B->B*C', 'B->B+B', 'B->B-B', 'B->x', 'B->y', 'B->z'],
    "ntn_list": ['A'],
    "truth_ode_format": ["y", "-{0}*y-{1}*x-{2}*x*x*x+{3}*cos({4}*z)", "1"],
    "purification_threshold": 0.05,
}


class Duffing(ODEDataset):
    params_config = PARAMS_CONFIG
    
    def __init__(self, args):
        super(Duffing, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        delta, alpha, beta, gamma, omega = iter(self.params[env_id])
        x1, x2, x3 = x
        dy = np.asarray([
            x2,
            -delta * x2 - alpha * x1 - beta * x1 ** 3 + gamma * np.cos(omega * x3),
            1.0,
        ])
        return dy


if __name__ == "__main__":
    pass
