
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "Fluid_Flow",
    "env_max": 10,
    "ode_dim": 3,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [0.10, 1.00, -0.10, 10.0],
    "default_params_list": [
        [0.10, 1.00, -0.10, 10.0],
        [0.12, 1.20, -0.20, 10.2],
        [0.13, 1.30, -0.30, 10.3],
        [0.14, 1.40, -0.40, 10.4],
        [0.15, 1.50, -0.50, 10.5],
        [0.16, 1.60, 0.00, 10.6],
        [0.17, 1.25, 0.10, 10.7],
        [0.18, 0.90, -0.15, 11.0],
        [0.19, 0.80, -0.14, 10.9],
        [0.11, 0.70, -0.18, 9.5],
    ],
    "random_y0_base":  [1, 1, 1],
    "default_y0_list": [
        [1, 1, 1],
        [1.1, 1.1, 1.1],
        [1.2, 1.3, 1.2],
        [1.3, 1.4, 1.5],
        [2, 2, 2],
        [2.1, 2.5, 2.9],
        [2.2, 2.1, 2.8],
        [2.5, 2.4, 3.9],
        [2.9, 3.6, 3.0],
        [3, 3.2, 2.2],
    ],
    "dt": 0.01,
    "t_min": 0,
    "t_max": 10,
    "curve_names": ["x", "y", "z"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)', 
                'A->x', 'A->y','A->z'],
    "ntn_list": ['A'],
    "truth_ode_format": [
        "{0}*x-{1}*y+{2}*x*z",
        "{1}*x+{0}*y+{2}*y*z",
        "-{3}*(z-x**2-y**2)",
    ]
}


class FluidFlow(ODEDataset):
    params_config = PARAMS_CONFIG
    
    def __init__(self, args):
        super(FluidFlow, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        mu, omega, a, lam = iter(self.params[env_id])
        dy = np.asarray([
            mu * x[0] - omega * x[1] + a * x[0] * x[2],
            omega * x[0] + mu * x[1] + a * x[1] * x[2],
            - lam * (x[2] - x[0] ** 2 - x[1] ** 2),
        ])
        return dy


if __name__ == "__main__":
    pass
