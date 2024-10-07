import numpy as np
import pandas as pd
import time
import os
import json
import argparse
from functools import partial
from score import simplify_eq, score_with_est
from spl_base import SplBase
from spl_task_utils import *
# from invariant_physics.spl import SplBase, score_with_est,
from invariant_physics.dataset import evaluate_trajectory_rmse, get_dataset, load_argparse, simplify_and_replace_constants, judge_expression_equal, check_existing_record

from utils import extract, get_now_string, setup_seed, remove_constant


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def run_spl(args, task, task_ode_num, num_run, transplant_step, data_dir='data/', max_len = 50, eta = 0.9999,
            max_module_init = 10, num_aug = 5, exp_rate = 1/np.sqrt(2), num_transplant = 20, 
            norm_threshold=1e-5, count_success = True):
    """
    Executes the main training loop of Symbolic Physics Learner.
    
    Parameters
    ----------
    task : String object.
        benchmark task name. 
    num_run : Int object.
        number of runs performed.
    transplant_step : Int object.
        number of iterations simulated for training between two transplantations. 
    data_dir : String object.
        directory of training data samples. 
    max_len : Int object.
        maximum allowed length (number of production rules ) of discovered equations.
    eta : Int object.
        penalty factor for rewarding. 
    max_module_init : Int object.
        initial maximum length for module transplantation candidates. 
    num_aug : Int object.
        number of trees for module transplantation. 
    exp_rate : Int object.
        initial exploration rate. 
    num_transplant : Int object.
        number of transplantation candidate update performed throughout traning. 
    norm_threshold : Float object.
        numerical error tolerance for norm calculation, a very small value. 
    count_success : Boolean object. 
        if success rate is recorded. 
        
    Returns
    -------
    all_eqs: List<Str>
        discovered equations. 
    success_rate: Float
        success rate of all runs performed. 
    all_times: List<Float>
        runtimes for successful runs. 
    """

    if check_existing_record(
        task_date=args.record_task_date,
        ode_name=args.task,
        n_dynamic=args.n_dynamic,
        noise_ratio=args.noise_ratio,
        task_ode_num=args.task_ode_num,
        env_id=args.env_id,
        seed=args.seed,
    ):
        print(f"Skipped Task: n_dynamic={args.n_dynamic}, noise_ratio={args.noise_ratio}, task_ode_num={args.task_ode_num}, env_id={args.env_id}, seed={args.seed}")
        return None, None, None

    
    return None, None, None


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # output_folder = 'results_dsr/'  ## directory to save discovered results
    # save_eqs = True                 ## if true, discovered equations are saved to "output_folder" dir
    #
    # parser = argparse.ArgumentParser(description='make_data')
    # parser.add_argument(
    #     '--task',
    #     default='Lotka_Volterra',
    #     type=str, help="""please select the benchmark task from the list
    #                       [Lotka-Volterra
    #                       go, # Glycolytic Oscillator
    #                       ]""")
    # parser.add_argument(
    #     "--task_ode_num",
    #     type=int,
    #     default=1, help="ODE # in current task, e.g. for Lotka-Volterra, 1 is for dx, 2 for dy")
    # parser.add_argument(
    #     '--num_env',
    #     default=-1,
    #     type=int, help='number of environments being used for scoring, default 4')
    # # parser.add_argument(
    # #     '--eta',
    # #     default=0.9999,
    # #     type=float, help='eta, parsimony coefficient, defaul 0.9999')
    # parser.add_argument(
    #     '--output_dir',
    #     default='results',
    #     type=str, help="""directory to store log and Monte Carlo trees""")
    # parser.add_argument(
    #     '--max_added_grammar_count',
    #     default=2,
    #     type=int, help='number of grammars can be inserted at once (in forced nodes), 0 means no insertion allowed')
    # parser.add_argument(
    #     "--force",
    #     type=str2bool,
    #     default=False, help="whether to force simplified nodes back into the tree or not")
    # parser.add_argument(
    #     "--use_new_reward",
    #     type=int,
    #     default=0, help="0: old score, 1: new score-min, 2: new score-mean")
    # parser.add_argument(
    #     "--reward_rescale",
    #     type=str2bool,
    #     default=False, help="whether or not to use rescale in the reward function")
    # parser.add_argument(
    #     '--data_dir',
    #     default='./data',
    #     type=str, help="""directory to datasets""")
    # parser.add_argument(
    #     '--error_tolerance',
    #     default=0.99,
    #     type=float, help='error_tolerance for reward functions 1 and 2, default 0.99')
    # parser.add_argument(
    #     "--num_run",
    #     type=int,
    #     default=1, help="Number of tests to run")
    # parser.add_argument(
    #     "--transplant_step",
    #     type=int,
    #     default=500, help="Number of MCTS iterations per transplant")
    # parser.add_argument(
    #     "--loss_func",
    #     type=str,
    #     choices=["VF", "L2"],
    #     default="L2", help="loss function: L2 or VF")
    # parser.add_argument(
    #     '--combine_operator',
    #     default='min',
    #     type=str, help="""please select which operator used to combine rewards from different environments:
    #                       [min
    #                        average,
    #                       ]""")
    # parser.add_argument(
    #     '--min_lam_diff',
    #     default=0.,
    #     type=float, help='minimum weight for difference/residual reward in reward_function 2, default 0.0')
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=420, help="Random seed used")
    # parser.add_argument("--noise_ratio", type=float, default=0.00, help="noise ratio")
    # parser.add_argument("--resume", type=int, default=0, help="resume (1) or not (0)")
    # parser.add_argument("--train_test_total", type=str, default="500",
    #                     help="num_train+num_test. E.g., '500', '500/400/300/200/100'. If you want to specify the total of different environment, use '/' to split")
    # parser.add_argument("--dataset_sparse", type=str, default="sparse", choices=["sparse", "dense"],
    #                     help="sparse or dense")
    # parser.add_argument("--save_figure", type=int, default=0, help="save figure or not")
    # parser.add_argument("--dataset_gp", type=int, default=0, choices=[0, 1], help="Gaussian Process or not")
    # parser.add_argument('--main_path', type=str, default="./", help="""directory to the main path""")
    # parser.add_argument("--env_id", type=int, default=-1, help="0,1,2,3,4")
    # parser.add_argument('--eta', default=0.99, type=float, help='eta, parsimony coefficient, default 0.99')
    # parser.add_argument('--integrate_method', type=str, default="ode_int", choices=["ode_int", "solve_ivp"], help="""integrate_method""")
    # parser.add_argument("--train_ratio", type=float, default=0.8, help="train_ratio")
    # parser.add_argument("--test_ratio", type=float, default=0.2, help="test_ratio")


    # args = parser.parse_args()

    args, parser = load_argparse()
    print(f"timestring: {args.timestring}")
    print(f"main_path: {args.main_path}")

    RULEMAP = ['A->(A+A)', 'A->(A-A)', 'A->(A*A)', 'A->(A/A)', 'A->(A*C)',
                         'A->x', 'A->y']
    NTN_LIST = ['A']
    task = args.task
    task_ode_num = args.task_ode_num
    eta = args.eta
    num_env = args.num_env
    # np.random.seed(args.seed)
    setup_seed(seed=args.seed)
    # print("="*30)
    # print(f"Task: {task} #{task_ode_num}\n")
    # print(f"# environments: {num_env}")
    # print(f"Eta={eta}, reward function:{args.use_new_reward} {args.combine_operator}")
    # print(f"Full args:")
    # print(args)
    # print("="*30)
    all_eqs, _, _ = run_spl(args, task, task_ode_num,
                            num_run=args.num_run,
                            max_len=50,
                            eta=1 - 1e-3,
                            max_module_init=20,
                            num_transplant=3,
                            num_aug=0,
                            transplant_step=args.transplant_step,
                            count_success=True,
                            data_dir='data/',
                            )




