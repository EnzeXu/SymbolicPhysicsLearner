
from .ode import *
from ._utils import params_random, params_default

PARAMS_CONFIG = {
    "task": "MAP_Bistable",
    "env_max": 10,
    "ode_dim": 11,
    "ode_dim_function": 11,
    "params_strategy_list": ["default", "random"],
    "params": {
        "default": params_default,
        "random": params_random,
    },
    "random_params_base": [0.00275, 2.5, 0.025, 0.00445, 37.5, 0.00625, 0.23, 0.0014, 1.25, 0.215, 0.0001525],
    "default_params_list": [
        [0.00275, 2.5, 0.025, 0.00445, 37.5, 0.00625, 0.23, 0.0014, 1.25, 0.215, 0.0001525],
    ],
    "random_y0_base": [9000, 900, 0, 0, 0, 0, 1800, 0, 0, 0, 0],
    "default_y0_list": [
        [9000, 900, 0, 0, 0, 0, 1800, 0, 0, 0, 0],
    ],
    "dt": 0.005,
    "t_min": 0,
    "t_max": 100,
    "curve_names": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
    "rule_map": ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->A*C',
                 'A->a', 'A->b', 'A->c', 'A->d', 'A->e', 'A->f', 'A->g', 'A->h', 'A->i', 'A->j', 'A->k',
                 'A->C'],
    "ntn_list": ['A'],
    "truth_ode_format": [
        "- {0} * a * b + {1} * c + {9} * j - {10} * a * g",
        "- {0} * a * b + {1} * c + {2} * c - {3} * d * b + {1} * e + {4} * e",
        "{0} * a * b - {1} * c - {2} * c",
        "{2} * c - {3} * d * b + {1} * e + {1} * i - {7} * d * g - {7} * d * g + {1} * k",
        "{3} * d * b - {1} * e - {4} * e",
        "{4} * e - {5} * f * g + {1} * h",
        "- {5} * f * g + {1} * h + {1} * i - {7} * d * g - {7} * d * g + {1} * k + {9} * j - {10} * a * g",
        "{5} * f * g - {1} * h - {6} * h",
        "{6} * h - {1} * i + {7} * d * g",
        "{8} * k - {9} * j + {10} * a * g",
        "{7} * d * g - {1} * k - {8} * k",
    ],
    "purification_threshold": 0.001,
}


class MAPBistable(ODEDataset):
    params_config = PARAMS_CONFIG

    def __init__(self, args):
        super(MAPBistable, self).__init__(args, PARAMS_CONFIG)

    def _func(self, x, t, env_id):
        # beta, n, rho, gamma = iter(self.params[env_id])
        assert len(self.params[env_id]) == 11
        k1, k1p, k2, k3, k4, h1, h2, h4, h5, h6, h6p = iter(self.params[env_id])
        dy = np.asarray([
            - k1*x[0]*x[1] + k1p*x[2] + h6*x[9] - h6p*x[0]*x[6],
            - k1*x[0]*x[1] + k1p*x[2] + k2*x[2] - k3*x[3]*x[1] + k1p*x[4] + k4*x[4],
            k1*x[0]*x[1] - k1p*x[2] - k2*x[2],
            k2*x[2] - k3*x[3]*x[1] + k1p*x[4] + k1p*x[8] - h4*x[3]*x[6] - h4*x[3]*x[6] + k1p*x[10],
            k3*x[3]*x[1] - k1p*x[4] - k4*x[4],
            k4 * x[4] - h1 * x[5] * x[6] + k1p * x[7],
            - h1*x[5]*x[6] + k1p*x[7] + k1p*x[8] - h4*x[3]*x[6] - h4*x[3]*x[6] + k1p*x[10] + h6*x[9] - h6p*x[0]*x[6],
            h1 * x[5] * x[6] - k1p * x[7] - h2 * x[7],
            h2 * x[7] - k1p * x[8] + h4 * x[3] * x[6],
            h5 * x[10] - h6 * x[9] + h6p * x[0] * x[6],
            h4 * x[3] * x[6] - k1p * x[10] - h5 * x[10],
        ])
        return dy


if __name__ == "__main__":
    pass
