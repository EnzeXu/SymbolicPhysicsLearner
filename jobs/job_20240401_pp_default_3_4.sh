#!/bin/bash
for dataset in "default_3"
do
    for model in "Lotka_Volterra" #"Lotka_Volterra" "Lorenz" "SIR" "Fluid_Flow"
    do
        for seed in {0..2}
        do
            for eq_id in "1" "2"
            do
                for noise_ratio in "0.004" #"0.000" "0.001" "0.002" "0.004" "0.008" "0.016" "0.032" "0.064" "0.128" "0.256" "0.512" "1.024"
                do
                    for env_id in "0" "1" "2" "3" "4"
                    do
                        echo "python spl_train.py --task ${model} --num_env 5 --use_new_reward 0 --num_run 1 --noise_ratio ${noise_ratio} --seed ${seed} --task_ode_num ${eq_id} --train_test_total ${dataset}"
                        python spl_train.py --task ${model} --num_env 5 --use_new_reward 0 --num_run 1 --noise_ratio ${noise_ratio} --seed ${seed} --task_ode_num ${eq_id} --train_test_total ${dataset}
                    done
                done
            done   
        done
    done
done
