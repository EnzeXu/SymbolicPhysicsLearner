.PHONY: test

test:
	python -u spl_train.py --task Lotka_Volterra --num_env 5 --env_id 0 --use_new_reward 0 --num_run 1 --noise_ratio 0.00 --seed 0 --task_ode_num 1 --transplant_step 500 --eta 0.9999  --combine_operator average --n_dynamic 10/10/10/10/10 --num_transplant 1 --n_data_samples 400 --extract_csv 1