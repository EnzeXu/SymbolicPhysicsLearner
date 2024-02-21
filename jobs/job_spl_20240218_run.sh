#!/bin/bash
for noise_ratio in "0.000" "0.001" "0.002" "0.004" "0.008" "0.016" "0.032" "0.064" "0.128" "0.256" "0.512" "1.024"
do
    for seed in {0..11}
    do
        for task in "Lotka_Volterra" # "Lotka_Volterra" "SIR" "Lorenz"
        do
            for eq_id in "1" "2"
            do
                for env_id in "0" "1" "2" "3" "4"
                do
                    echo "python spl_train.py --task ${task} --num_env 5 --num_run 1 --task_ode_num ${eq_id} --env_id ${env_id} --seed ${seed} --noise_ratio ${noise_ratio}"
                    python spl_train.py --task ${task} --num_env 5 --num_run 1 --task_ode_num ${eq_id} --env_id ${env_id} --seed ${seed} --noise_ratio ${noise_ratio}
                done
            done
        done
    done

    for seed in {0..11}
    do
        for task in "SIR" # "Lotka_Volterra" "SIR" "Lorenz"
        do
            for eq_id in "1" "2" "3"
            do
                for env_id in "0" "1" "2" "3" "4"
                do
                    echo "python spl_train.py --task ${task} --num_env 5 --num_run 1 --task_ode_num ${eq_id} --env_id ${env_id} --seed ${seed} --noise_ratio ${noise_ratio}"
                    python spl_train.py --task ${task} --num_env 5 --num_run 1 --task_ode_num ${eq_id} --env_id ${env_id} --seed ${seed} --noise_ratio ${noise_ratio}
                done
            done
        done
    done

    for seed in {0..11}
    do
        for task in "Lorenz" # "Lotka_Volterra" "SIR" "Lorenz"
        do
            for eq_id in "1" "2" "3"
            do
                for env_id in "0" "1" "2" "3" "4"
                do
                    echo "python spl_train.py --task ${task} --num_env 5 --num_run 1 --task_ode_num ${eq_id} --env_id ${env_id} --seed ${seed} --noise_ratio ${noise_ratio}"
                    python spl_train.py --task ${task} --num_env 5 --num_run 1 --task_ode_num ${eq_id} --env_id ${env_id} --seed ${seed} --noise_ratio ${noise_ratio}
                done
            done
        done
    done
done
