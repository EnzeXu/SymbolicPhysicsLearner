#!/bin/bash


for model in "Lotka_Volterra"
do
    echo "# model=${model}"
    for noise_ratio in "0.000" "0.002" "0.004" "0.006" "0.008" "0.010"
    do
        echo "## noise_ratio=${noise_ratio}"
        for eq_id in "1"
        do
            for seed in "0"
            do
                echo "###### seed=${seed}"
                python make_datasets.py --task ${model} --num_env 5 --noise_ratio ${noise_ratio} --seed ${seed} --task_ode_num ${eq_id} --train_test_total 500 --transplant_step 500
            done
        done
    done
done



for model in "SIR"
do
    echo "# model=${model}"
    for noise_ratio in "0.000" "0.002" "0.004" "0.006" "0.008" "0.010"
    do
        echo "## noise_ratio=${noise_ratio}"
        for eq_id in "1"
        do
            for seed in "0"
            do
                echo "###### seed=${seed}"
                python make_datasets.py --task ${model} --num_env 5 --noise_ratio ${noise_ratio} --seed ${seed} --task_ode_num ${eq_id} --train_test_total 500 --transplant_step 500
            done
        done
    done
done

for model in "Lorenz"
do
    echo "# model=${model}"
    for noise_ratio in "0.000" "0.002" "0.004" "0.006" "0.008" "0.010"
    do
        echo "## noise_ratio=${noise_ratio}"
        for eq_id in "1"
        do
            for seed in "0"
            do
                echo "###### seed=${seed}"
                python make_datasets.py --task ${model} --num_env 5 --noise_ratio ${noise_ratio} --seed ${seed} --task_ode_num ${eq_id} --train_test_total 125 --transplant_step 500 --dataset_sparse dense
            done
        done
    done
done