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


    func_score = score_with_est
    func_score = partial(func_score, task_ode_num=task_ode_num)
    
    ## define production rules and non-terminal nodes. 
    grammars = rule_map[task]
    # print(rule_map.keys())
    nt_nodes = ntn_map[task]


    ## read training and testing data
    # train_sample = pd.read_csv(data_dir + task + '_train.csv', header=None).to_numpy().T
    # test_sample = pd.read_csv(data_dir + task + '_test.csv', header=None).to_numpy().T

    noise_ratio = args.noise_ratio
    env_id = args.env_id

    log_start_time = get_now_string()
    log_save_folder = f"logs/{task}/"
    log_summary_save_folder = f"logs/summary/"
    log_save_detail_path = f"logs/{task}/{log_start_time}.csv"
    # log_save_term_trace_path = f"logs/{task}/{log_start_time}_term_trace.png"
    # log_save_test_dic_full_path = f"logs/{task}/{log_start_time}_full_test.pkl"
    log_path = f"{log_summary_save_folder}/logs_{task}.csv"

    if not os.path.exists(log_save_folder):
        os.makedirs(log_save_folder)
    if not os.path.exists(log_summary_save_folder):
        os.makedirs(log_summary_save_folder)
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(
                f"start_time,status,end_time,task,num_run,num_env,success_boolean,truth_ode,prediction_ode,reward_func_id,loss_func,noise_ratio,task_ode_num,dataset_sparse,env_id,seed\n")
    with open(log_path, "a") as f:
        f.write(f"{log_start_time},Begin,{None},{task},{num_run},{num_env},{None},{None},{None},{args.use_new_reward},{args.loss_func},{noise_ratio:.6f},{task_ode_num},{args.dataset_sparse},{env_id},{args.seed}\n")



    train_sample = pd.read_csv(os.path.join(data_dir, task, f"{noise_ratio:.3f}", f'{task}_train_{env_id}.csv'))
    test_sample = pd.read_csv(os.path.join(data_dir, task, f"{noise_ratio:.3f}", f'{task}_test_{env_id}.csv'))

    with open(f'{data_dir}/{task}/{noise_ratio:.3f}/{task}_info.json', 'r') as f:
        info = json.load(f)

    num_success = 0
    all_times = []
    all_eqs = []
    
    ## number of module max size increase after each transplantation 
    module_grow_step = (max_len - max_module_init) / num_transplant

    log_truth_ode_list = info['log_truth_ode_list']
    truth_ode = log_truth_ode_list[env_id][task_ode_num - 1]
    _, truth_ode_terms, _ = extract(truth_ode)
    truth_ode_terms = remove_constant(truth_ode_terms)

    for i_test in range(num_run):

        print("test", i_test)
        best_solution = ('nothing', 0)

        exploration_rate = exp_rate
        max_module = max_module_init
        reward_his = []
        best_modules = []
        aug_grammars = []

        start_time = time.time()
        discovery_time = 0

        # print("data_sample", train_sample)
        # print("base_grammars", grammars)
        # print("aug_grammars", aug_grammars)
        # print("nt_nodes", nt_nodes)
        # print("max_len", max_len)
        # print("max_module", max_module)
        # print("aug_grammars_allowed", num_aug)
        # print("func_score", score_with_est)
        # print("exploration_rate", exploration_rate)
        # print("eta", eta)


        for i_itr in range(num_transplant):

            spl_model = SplBase(data_sample = train_sample,
                                base_grammars = grammars, 
                                aug_grammars = aug_grammars, 
                                nt_nodes = nt_nodes, 
                                max_len = max_len, 
                                max_module = max_module,
                                aug_grammars_allowed = num_aug,
                                func_score = func_score,
                                exploration_rate = exploration_rate, 
                                eta = eta)


            histroy, current_solution, good_modules = spl_model.run(transplant_step,
                                                              num_play=10, 
                                                              print_flag=True)
            # print("histroy", histroy)

            end_time = time.time() - start_time

            if not best_modules:
                best_modules = good_modules
            else:
                best_modules = sorted(list(set(best_modules + good_modules)), key = lambda x: x[1])
            aug_grammars = [x[0] for x in best_modules[-num_aug:]]
            
            # print([simplify_eq(x[2]) for x in best_modules[-num_aug:]])

            reward_his.append(best_solution[1])

            if current_solution[1] > best_solution[1]:
                best_solution = current_solution
            # print(best_solution)
            max_module += module_grow_step
            exploration_rate *= 5

            # check if solution is discovered. Early stop if it is. 
            # test_score = score_with_est(simplify_eq(best_solution[0]), 0, test_sample, eta = eta)[0]
            test_score = func_score(simplify_eq(best_solution[0]), 0, test_sample, eta=eta)[0]
            if test_score >= 1 - norm_threshold:
                num_success += 1
                if discovery_time == 0:
                    discovery_time = end_time
                    all_times.append(discovery_time)
                break

        all_eqs.append(simplify_eq(best_solution[0]))
        best_res = simplify_eq(best_solution[0])
        _, best_res_terms, _ = extract(str(best_res))
        # print(best_res_terms)
        best_res_terms = remove_constant(best_res_terms)
        # print(best_res_terms)
        # print('\n{} tests complete after {} iterations.'.format(i_test+1, i_itr+1))
        print(f"truth: {truth_ode}")
        print(f"truth terms: {str(truth_ode_terms)}")
        print('best solution: {}'.format(best_res))
        print('best solution terms: {}'.format(str(best_res_terms)))
        # print('test score: {}'.format(test_score))

        # print()
        log_end_time = get_now_string()
        print(f"success: {int(str(truth_ode_terms)==str(best_res_terms))}")
        with open(log_path, "a") as f:
            f.write(f"{log_start_time},End,{log_end_time},{task},{num_run},{num_env},{int(str(truth_ode_terms)==str(best_res_terms))},{truth_ode.replace(',',';')},{best_res.replace(',',';')},{args.use_new_reward},{args.loss_func},{noise_ratio:.6f},{task_ode_num},{args.dataset_sparse},{args.env_id},{args.seed}\n")

    success_rate = num_success / num_run
    # if count_success:
    #     print(f'success rate = num_success / num_run = {num_success} / {num_run} =', success_rate)
    
    return all_eqs, success_rate, all_times


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    output_folder = 'results_dsr/'  ## directory to save discovered results
    save_eqs = True                 ## if true, discovered equations are saved to "output_folder" dir

    parser = argparse.ArgumentParser(description='make_data')
    parser.add_argument(
        '--task',
        default='Lotka_Volterra',
        type=str, help="""please select the benchmark task from the list 
                          [Lotka-Volterra
                          go, # Glycolytic Oscillator
                          ]""")
    parser.add_argument(
        "--task_ode_num",
        type=int,
        default=1, help="ODE # in current task, e.g. for Lotka-Volterra, 1 is for dx, 2 for dy")
    parser.add_argument(
        '--num_env',
        default=-1,
        type=int, help='number of environments being used for scoring, default 4')
    parser.add_argument(
        '--eta',
        default=0.9999,
        type=float, help='eta, parsimony coefficient, defaul 0.9999')
    parser.add_argument(
        '--output_dir',
        default='results',
        type=str, help="""directory to store log and Monte Carlo trees""")
    parser.add_argument(
        '--max_added_grammar_count',
        default=2,
        type=int, help='number of grammars can be inserted at once (in forced nodes), 0 means no insertion allowed')
    parser.add_argument(
        "--force",
        type=str2bool,
        default=False, help="whether to force simplified nodes back into the tree or not")
    parser.add_argument(
        "--use_new_reward",
        type=int,
        default=0, help="0: old score, 1: new score-min, 2: new score-mean")
    parser.add_argument(
        "--reward_rescale",
        type=str2bool,
        default=False, help="whether or not to use rescale in the reward function")
    parser.add_argument(
        '--data_dir',
        default='./data',
        type=str, help="""directory to datasets""")
    parser.add_argument(
        '--error_tolerance',
        default=0.99,
        type=float, help='error_tolerance for reward functions 1 and 2, default 0.99')
    parser.add_argument(
        "--num_run",
        type=int,
        default=1, help="Number of tests to run")
    parser.add_argument(
        "--transplant_step",
        type=int,
        default=500, help="Number of MCTS iterations per transplant")
    parser.add_argument(
        "--loss_func",
        type=str,
        choices=["VF", "L2"],
        default="L2", help="loss function: L2 or VF")
    parser.add_argument(
        '--combine_operator',
        default='min',
        type=str, help="""please select which operator used to combine rewards from different environments: 
                          [min
                           average, 
                          ]""")
    parser.add_argument(
        '--min_lam_diff',
        default=0.,
        type=float, help='minimum weight for difference/residual reward in reward_function 2, default 0.0')
    parser.add_argument(
        "--seed",
        type=int,
        default=420, help="Random seed used")
    parser.add_argument("--noise_ratio", type=float, default=0.00, help="noise ratio")
    parser.add_argument("--resume", type=int, default=0, help="resume (1) or not (0)")
    parser.add_argument("--train_test_total", type=int, default=250, help="num_train+num_test")
    parser.add_argument("--env_id", type=int, default=-1, help="0,1,2,3,4")
    parser.add_argument("--dataset_sparse", type=str, default="sparse", choices=["sparse", "dense"], help="sparse or dense")
    # parser.add_argument(
    #     "--log_suffix",
    #     type=str,
    #     default="", help="log name suffix")
    args = parser.parse_args()

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
                            num_transplant=1,
                            num_aug=0,
                            transplant_step=args.transplant_step,
                            count_success=True,
                            data_dir='data/',
                            )




