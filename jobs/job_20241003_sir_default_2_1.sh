#!/bin/bash
for n_dynamic in "default_2"
do
    for model in "SIR" #"Lotka_Volterra" "Lorenz" "SIR" "Fluid_Flow"
    do
        for seed in {0..19}
        do
            for eq_id in "1" "2" "3"
            do
                for noise_ratio in "0.00" #"0.000" "0.001" "0.002" "0.004" "0.008" "0.016" "0.032" "0.064" "0.128" "0.256" "0.512" "1.024"
                do
                    for env_id in "0" "1" "2" "3" "4"
                    do
                        taskset -c 12-31 timestring=$(taskset -c 12-31 python shell_timestring.py)
                        taskset -c 12-31 output_path="outputs/${model}_${timestring}.txt"
                        taskset -c 12-31 echo "python -u spl_train.py --task ${model} --num_env 5 --env_id ${env_id} --use_new_reward 0 --num_run 1 --noise_ratio ${noise_ratio} --seed ${seed} --task_ode_num ${eq_id} --transplant_step 500 --eta 0.9999  --combine_operator average --n_dynamic ${n_dynamic} --num_transplant 1 --n_data_samples 400 --extract_csv 1"
                        taskset -c 12-31 echo "python -u spl_train.py --task ${model} --num_env 5 --env_id ${env_id} --use_new_reward 0 --num_run 1 --noise_ratio ${noise_ratio} --seed ${seed} --task_ode_num ${eq_id} --transplant_step 500 --eta 0.9999  --combine_operator average --n_dynamic ${n_dynamic} --num_transplant 1 --n_data_samples 400 --extract_csv 1" >> ${output_path}
                        taskset -c 12-31 python -u spl_train.py --task ${model} --num_env 5 --env_id ${env_id} --use_new_reward 0 --num_run 1 --noise_ratio ${noise_ratio} --seed ${seed} --task_ode_num ${eq_id} --transplant_step 500 --eta 0.9999  --combine_operator average --n_dynamic ${n_dynamic} --num_transplant 1 --n_data_samples 400 --extract_csv 1 >> ${output_path}
                    done
                done
            done   
        done
    done
done
