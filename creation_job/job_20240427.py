# template = """#!/bin/bash
# for noise_ratio in "{0}"
# do
#     for seed in {{0..11}}
#     do
#         for task in "Lotka_Volterra" # "Lotka_Volterra" "SIR" "Lorenz"
#         do
#             for eq_id in "1" "2"
#             do
#                 for env_id in "0" "1" "2" "3" "4"
#                 do
#                     echo "python spl_train.py --task ${{task}} --num_env 5 --num_run 1 --task_ode_num ${{eq_id}} --env_id ${{env_id}} --seed ${{seed}} --noise_ratio ${{noise_ratio}}"
#                     python spl_train.py --task ${{task}} --num_env 5 --num_run 1 --task_ode_num ${{eq_id}} --env_id ${{env_id}} --seed ${{seed}} --noise_ratio ${{noise_ratio}}
#                 done
#             done
#         done
#     done
#
#     for seed in {{0..11}}
#     do
#         for task in "SIR" # "Lotka_Volterra" "SIR" "Lorenz"
#         do
#             for eq_id in "1" "2" "3"
#             do
#                 for env_id in "0" "1" "2" "3" "4"
#                 do
#                     echo "python spl_train.py --task ${{task}} --num_env 5 --num_run 1 --task_ode_num ${{eq_id}} --env_id ${{env_id}} --seed ${{seed}} --noise_ratio ${{noise_ratio}}"
#                     python spl_train.py --task ${{task}} --num_env 5 --num_run 1 --task_ode_num ${{eq_id}} --env_id ${{env_id}} --seed ${{seed}} --noise_ratio ${{noise_ratio}}
#                 done
#             done
#         done
#     done
#
#     for seed in {{0..11}}
#     do
#         for task in "Lorenz" # "Lotka_Volterra" "SIR" "Lorenz"
#         do
#             for eq_id in "1" "2" "3"
#             do
#                 for env_id in "0" "1" "2" "3" "4"
#                 do
#                     echo "python spl_train.py --task ${{task}} --num_env 5 --num_run 1 --task_ode_num ${{eq_id}} --env_id ${{env_id}} --seed ${{seed}} --noise_ratio ${{noise_ratio}}"
#                     python spl_train.py --task ${{task}} --num_env 5 --num_run 1 --task_ode_num ${{eq_id}} --env_id ${{env_id}} --seed ${{seed}} --noise_ratio ${{noise_ratio}}
#                 done
#             done
#         done
#     done
# done
# """
#
# for i, j in zip(range(12), ["0.000", "0.001", "0.002", "0.004", "0.008", "0.016", "0.032", "0.064", "0.128", "0.256", "0.512", "1.024"]):
#     with open(f"jobs/job_20240218_run_{i + 1}.sh", "w") as f:
#         f.write(template.format(j))
#
# # short = "pp"

short_dic = {
    "pp": "Lotka_Volterra",
    "lorenz": "Lorenz",
    "sir": "SIR",
}

eq_id_dic = {
    "pp": '"1" "2"',
    "lorenz": '"1" "2" "3"',
    "sir": '"1" "2" "3"',
}

template = """#!/bin/bash
for dataset in "{1}"
do
    for model in "{0}" #"Lotka_Volterra" "Lorenz" "SIR" "Fluid_Flow"
    do
        for seed in {{0..11}}
        do
            for eq_id in {3}
            do
                for noise_ratio in "{2}" #"0.000" "0.001" "0.002" "0.004" "0.008" "0.016" "0.032" "0.064" "0.128" "0.256" "0.512" "1.024"
                do
                    for env_id in "0"
                    do
                        echo "python spl_train.py --task ${{model}} --num_env 1 --env_id ${{env_id}} --use_new_reward 0 --num_run 1 --noise_ratio ${{noise_ratio}} --seed ${{seed}} --task_ode_num ${{eq_id}} --train_test_total ${{dataset}} --transplant_step 500 --train_ratio 1.00 --test_ratio 1.00 --integrate_method ode_int "
                        python spl_train.py --task ${{model}} --num_env 1 --env_id ${{env_id}} --use_new_reward 0 --num_run 1 --noise_ratio ${{noise_ratio}} --seed ${{seed}} --task_ode_num ${{eq_id}} --train_test_total ${{dataset}} --transplant_step 500 --train_ratio 1.00 --test_ratio 1.00 --integrate_method ode_int 
                    done
                done
            done   
        done
    done
done
"""

launch_template = """#!/bin/bash
# Navigate to the workspace
cd /home/exu03/workspace/SymbolicPhysicsLearner

# Check if the tmux session exists
tmux has-session -t {0}_{1}_{2} 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s {0}_{1}_{2}

  tmux send-keys -t {0}_{1}_{2} "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t {0}_{1}_{2} "bash jobs/job_20240427_{0}_{1}_{2}.sh" C-m
  echo "Launched jobs/job_20240427_{0}_{1}_{2}.sh on tmux session {0}_{1}_{2}"
  # If you want to leave the session detached, remove the line below
  # tmux send-keys -t {0}"exit" C-m
else
  echo "Session '{0}_{1}_{2}' already exists. Attaching..."
  tmux attach -t {0}_{1}_{2}
  tmux send-keys -t {0}_{1}_{2} "source ../Invariant_Physics/venv/bin/activate" C-m
  tmux send-keys -t {0}_{1}_{2} "bash jobs/job_20240401_{0}_{1}_{2}.sh" C-m
fi
"""

launch_all_template = """#!/bin/bash
cd /home/exu03/workspace/SymbolicPhysicsLearner
{0}
"""
noise_list = ["0.000", "0.001", "0.002", "0.004", "0.008", "0.016", "0.032", "0.064", "0.128", "0.256", "0.0001", "0.00001"]

process_list_map = [[] for _ in range(len(noise_list))]
for k in range(0, 2, 1):
    size = ["500", "1000"][k]
    for short in ["pp"]:

        for i, j in zip(range(len(noise_list)), noise_list):
            with open(f"jobs/job_20240427_{short}_{size}_{i+1}.sh", "w") as f:
                f.write(template.format(short_dic[short], size, j, eq_id_dic[short]))
            with open(f"launch_job/process_20240427_{short}_{size}_{i+1}.sh", "w") as f:
                f.write(launch_template.format(short, size, i+1))
            process_list_map[i].append(f"launch_job/process_20240427_{short}_{size}_{i+1}.sh")

for i, j in zip(range(len(noise_list)), noise_list):
    with open(f"launch_job/launch_20240427_server_{i+1}.sh", "w") as f:
        print(f"In launch_job/launch_20240427_server_{i+1}.sh")
        all_process = "\n".join([
            f"bash {item}" for item in process_list_map[i]
        ])
        f.write(launch_all_template.format(
            all_process
        ))
    print(all_process)
    print()