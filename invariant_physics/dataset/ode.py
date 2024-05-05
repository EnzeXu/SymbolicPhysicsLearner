import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import odeint, solve_ivp
from tqdm import tqdm


from ._utils import sample_lhs, save_to_csv
from .gp import GPPCA0


class ODEDataset:
    def __init__(self, args, params_config):
        self.args = args
        self.setup_seed(self.args.seed)
        self.params_config = params_config
        self.train_test_total_list = self.args.train_test_total_list
        print(f"self.train_test_total_list: {self.train_test_total_list}")
        self.ode_dim = self.params_config["ode_dim"]
        self.ode_name = self.params_config["task"]
        self.params, self.y0 = self._get_ode_params_and_y0(self.args.num_env, self.args.params_strategy)
        self.params_eval = [0.0 for i in range(len(self.params_config["curve_names"]))]
        self.t_series_list = []
        # self.train_indices_list, self.test_indices_list = [], []
        self.num_train_list, self.num_test_list = [], []
        # self.num_train, self.num_test = None, None
        # self.n = int(self.params_config["t_max"] / self.params_config["dt"])
        self.dt = self.params_config["dt"]
        self._set_t()

        self.y = [np.zeros([self.train_test_total_list[i], self.ode_dim]) for i in range(self.args.num_env)]
        self.y_noise = [np.zeros([self.train_test_total_list[i], self.ode_dim]) for i in range(self.args.num_env)]
        self.dy_noise = [np.zeros([self.train_test_total_list[i], self.ode_dim]) for i in range(self.args.num_env)]


        # self._build()

    def _get_ode_params_and_y0(self, num_env: int, params_strategy: str):
        assert num_env <= self.params_config.get("env_max")
        assert params_strategy in self.params_config.get("params_strategy_list")
        params_func = self.params_config["params"][params_strategy]
        params = params_func(
            num_env=num_env,
            default_list=self.params_config["default_params_list"],
            base=self.params_config["random_params_base"],
            seed=self.args.seed,
            random_rate=0.1,)
        y0 = params_func(
            num_env=num_env,
            default_list=self.params_config["default_y0_list"],
            base=self.params_config["random_y0_base"],
            seed=self.args.seed,
            random_rate=0.1, )
        return params, y0

    def _func(self, x, t, env_id):
        raise NotImplemented

    def _func_solve_ivp(self, t, x, env_id):
        return self._func(x, t, env_id)

    def _set_t(self):
        assert self.args.sample_strategy in ["uniform", "lhs"]
        for i in range(self.args.num_env):
            if self.args.sample_strategy == "uniform":
                self.t_series_list.append(np.asarray([self.params_config["t_min"] + self.dt * j for j in range(self.train_test_total_list[i])]))
                    # np.linspace(self.params_config["t_min"], self.params_config["t_max"] - self.params_config["dt"], self.n)
            else:  # lhs
                self.t_series_list.append(sample_lhs(self.params_config["t_min"], self.params_config["t_max"], self.train_test_total_list[i]))
        # print(self.t_series)

    def build(self):
        # train_test_total = self.args.train_test_total
        self.num_train_list = [int(self.args.train_ratio * one_train_test_total) for one_train_test_total in self.args.train_test_total_list]
        # self.num_test = train_test_total - self.num_train
        # self.num_test_list = [one_train_test_total - one_num_train for one_train_test_total, one_num_train in zip(self.args.train_test_total_list, self.num_train_list)]
        self.num_test_list = [int(self.args.test_ratio * one_train_test_total) for one_train_test_total in self.args.train_test_total_list]
        # indices = np.arange(self.n)
        # # for one_num_train, one_num_test in zip(self.num_train_list, self.num_test_list):
        # #     assert one_num_train + one_num_test <= self.n
        # if self.args.train_sample_strategy == "uniform":
        #     assert self.args.dataset_sparse in ["sparse", "dense"]
        #     train_indices_list, test_indices_list = [], []
        #     for i in range(self.args.num_env):
        #         one_num_train, one_num_test = self.num_train_list[i], self.num_test_list[i]
        #         if self.args.dataset_sparse == "sparse":
        #             train_indices_list.append(indices[0::(int(self.n / 1.01) // one_num_train)][:one_num_train])
        #             if one_num_test > 0:
        #                 test_indices_list.append(indices[0::(int(self.n / 1.01) // one_num_test)][:one_num_test])
        #         else:
        #             sample_freq = int(0.03 / self.dt) if int(0.05 / self.dt) >= 1 else 1
        #             train_indices_list.append(indices[::sample_freq][:one_num_train])
        #             if one_num_test > 0:
        #                 test_indices_list.append(indices[::sample_freq][:one_num_test])
        # else:  # random
        #     train_indices_list, test_indices_list = [], []
        #     for i in range(self.args.num_env):
        #         one_num_train, one_num_test = self.num_train_list[i], self.num_test_list[i]
        #         np.random.shuffle(indices)
        #         train_indices_list.append(sorted(indices[:one_num_train]))
        #         if one_num_test > 0:
        #             test_indices_list.append(sorted(indices[one_num_train: one_num_train + one_num_test]))
        # self.train_indices_list = train_indices_list
        # self.test_indices_list = test_indices_list

        save_folder = os.path.join(self.args.main_path, self.args.data_dir, self.ode_name, self.args.time_string)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        print(f"ode_name: {self.ode_name}")
        print(f"num_env: {self.args.num_env}")
        print(f"t_min: {self.params_config['t_min']}, t_max: {self.params_config['t_max']} (not used), dt: {self.params_config['dt']}")
        print(f"train_test_total_list: {self.args.train_test_total_list}")
        for i in range(self.args.num_env):
            one_train_test_total = self.args.train_test_total_list[i]
            one_num_train = self.num_train_list[i]
            one_num_test = self.num_test_list[i]
            print(f"Environment {i:02d}: train set: {one_num_train} ({one_num_train / one_train_test_total * 100:.1f} %), test set: {one_num_test} ({one_num_test / one_train_test_total * 100:.1f} %), train_test_total: {one_train_test_total}, n_total: {self.train_test_total_list[i]}")
        # print(f"train set: {self.num_train} ({self.num_train / train_test_total * 100:.1f} %), test set: {self.num_test} ({self.num_test / train_test_total * 100:.1f} %), train_test_total: {train_test_total}, n_total: {self.n}")
        print(f"noise ratio: {self.args.noise_ratio}")
        print(f"save_folder: {save_folder}")

        self._save_task_info(save_folder)

        for i in range(self.args.num_env):
            print(f"Environment {i:02d}: params={[f'{item:.8f}' for item in self.params[i]]}, y0={[f'{item:.8f}' for item in self.y0[i]]}, truth = {[item.format(*self.params[i]) for item in (self.params_config['truth_ode_format'] if self.params_config.get('truth_ode_format') else [])]}")


        for i in tqdm(range(self.args.num_env)):
            print(f"integrate_method: {self.args.integrate_method}")
            assert self.args.integrate_method in ["ode_int", "solve_ivp"]
            if self.args.integrate_method == "ode_int":
                self.y[i] = odeint(self._func, self.y0[i], self.t_series_list[i], (i,))
            else:
                sol = solve_ivp(fun=self._func_solve_ivp, t_span=(self.t_series_list[i][0], self.t_series_list[i][-1]), y0=np.asarray(self.y0[i]), args=(i,), t_eval=self.t_series_list[i], method="RK45")
                self.y[i] = np.transpose(sol.y, (1, 0))
            # self.y_noise[i] = self.y[i] * (1 + 2.0 * self.args.noise_ratio * (-0.5 + np.random.random(self.y[i].shape)))
            # print("self.y_noise[i].shape", self.y_noise[i].shape)

            std_base = np.std(self.y[i], axis=0)
            noise_sigma = std_base * self.args.noise_ratio
            # noise_sigma = np.broadcast_to(noise_sigma, (2000, 2))
            # print("std_base.shape", std_base.shape)
            # print("noise shape: ", (noise_sigma * np.random.randn(*self.y_noise[i].shape)).shape)
            self.y_noise[i] = self.y[i] + noise_sigma * np.random.randn(*self.y[i].shape)

            y_noise_train = self.y_noise[i]
            y_noise_test = self.y_noise[i]
            raw_path = os.path.join(save_folder, f"{self.ode_name}_{i}_raw.csv")
            save_to_csv(raw_path, [self.t_series_list[i]] + [self.y_noise[i][:, j] for j in range(self.ode_dim)], ["t"] + self.params_config["curve_names"][:self.ode_dim])

            if self.args.save_figure:
                t_train, t_test = self.t_series_list[i], self.t_series_list[i]
                y_train, y_test = self.y[i], self.y[i]

                save_path = os.path.join(save_folder, f"{self.ode_name}_{i}.png")
                self._plot_dataset(save_path, self.t_series_list[i], t_train, t_test, self.y[i], y_train, y_test, y_noise_train, y_noise_test)

            for j in range(self.ode_dim):
                self.dy_noise[i][:, j] = np.gradient(self.y_noise[i][:, j], self.t_series_list[i])
                
            dy_path_train = os.path.join(save_folder, f"{self.ode_name}_train_{i}.csv")
            dy_path_test = os.path.join(save_folder, f"{self.ode_name}_test_{i}.csv")

            # Generate points based on Gaussian Process of observation
            if self.args.dataset_gp:
                # Generate with frequency = `freq` points per second
                freq = 50
                t_max = max(self.t_series_list[i])
                t_min = min(self.t_series_list[i])
                num_points = int(freq * (t_max-t_min))
                t_train = np.linspace(t_min, t_max, num=num_points)
                y_train_generated = np.zeros([num_points, self.ode_dim])
                dy_noise_train = np.zeros([num_points, self.ode_dim])
                for j in range(self.ode_dim):
                    pca = GPPCA0(y_noise_train[:, j].reshape(-1,1), self.t_series_list[i], noise_sigma[j], sigma_out=std_base[j])
                    y_train_generated[:, j] = pca.get_predictive(new_sample=1, t_new=t_train).reshape(-1)
                    y_train_mean = pca.get_predictive_mean(t_new=t_train).reshape(-1)
                    y_train_std = np.sqrt(np.diag(pca.get_X_cov(t_new=t_train)))
                    if self.args.save_figure:
                        save_path = os.path.join(save_folder, f"{self.ode_name}_GP_{i}.png")
                        self._plot_GP(save_path, t_train, y_train_generated[:, j], y_train_mean, y_train_std)
#                     dy_noise_train[:, j] = np.gradient(y_train_generated[:, j], t_train)
                y_noise_train = y_train_generated
            else:
                t_train = self.t_series_list[i]
                dy_noise_train = self.dy_noise[i][:, :]
            
            save_to_csv(dy_path_train, 
                        [t_train] + [y_noise_train[:, j] for j in range(self.ode_dim)] + [dy_noise_train[:, j] for j in range(self.ode_dim)], 
                        ["t"] + self.params_config["curve_names"][:self.ode_dim] + ['d'+name for name in self.params_config["curve_names"][:self.ode_dim]])
            save_to_csv(dy_path_test,
                        [self.t_series_list[i]] + [y_noise_test[:, j] for j in range(self.ode_dim)] + [self.dy_noise[i][:, j] for j in range(self.ode_dim)],
                        ["t"] + self.params_config["curve_names"][:self.ode_dim] + ['d'+name for name in self.params_config["curve_names"][:self.ode_dim]])

    def _plot_dataset(self, save_path, t, t_train, t_test, y, y_train, y_test, y_noise_train, y_noise_test):
        assert len(t_train) == len(y_train) == len(y_noise_train)
        assert len(t_test) == len(y_test) == len(y_noise_test)
        plt.figure(figsize=(16, 9))
        for i in range(self.ode_dim):
            plt.plot(t, y[:, i], label=f"cur-{i + 1}")
        for i in range(self.ode_dim):
            # plt.scatter(t_train, y_train[:, i], label=f"curve-{i + 1} (train)")
            plt.scatter(t_train, y_noise_train[:, i], s=10, label=f"cur-{i + 1} [train noise] [n={len(t_train)}]")
            # plt.scatter(t_test, y_test[:, i], label=f"curve-{i + 1} (test)")
            plt.scatter(t_test, y_noise_test[:, i], s=10, label=f"cur-{i + 1} [test noise] [n={len(t_test)}]")
        plt.xlabel('Time')
        plt.ylabel('Val')
        plt.legend()
        plt.grid()
        plt.savefig(save_path, dpi=300)
        plt.clf()
    
    def _plot_GP(self, save_path, t_train, y_train_gen, y_train_mean, y_train_std):
        assert len(t_train) == len(y_train_mean) == len(y_train_std)
        plt.figure(figsize=(16, 9))
        for i in range(self.ode_dim):
            plt.fill_between(t_train, y_train_mean - y_train_std, y_train_mean + y_train_std, color='gray', alpha=0.5, label='Standard Deviation')
            plt.plot(t_train, y_train_mean, label='Mean', color='blue')
            plt.plot(t_train, y_train_gen, '+g', label='Generated datapoints')
        plt.xlabel('Time')
        plt.ylabel('Val')
        plt.legend()
        plt.grid()
        plt.savefig(save_path, dpi=300)
        plt.clf()

    @staticmethod
    def setup_seed(seed=0):
        np.random.seed(seed)
        random.seed(seed)

    def _save_task_info(self, save_folder):
        info_path = os.path.join(save_folder, f"{self.ode_name}_info.json")
        # log_truth_ode_list[i_env][task_ode_num - 1]
        task_info = {
            "params": {
                str(env): [x for x in self.params[env]] for env in range(self.args.num_env)
            },
            "y0": {
                str(env): [x for x in self.y0[env]] for env in range(self.args.num_env)
            },
            "noise_ratio": self.args.noise_ratio,
            "log_truth_ode_list": []
        }
        print(task_info)
        for i in range(self.args.num_env):
            log_truth_ode = [item.format(*task_info["params"][str(i)]) for item in
                             (self.params_config['truth_ode_format'] if self.params_config.get('truth_ode_format') else [])]
            task_info["log_truth_ode_list"].append(log_truth_ode)
        with open(info_path, 'w') as f:
            json.dump(task_info, f, sort_keys=True, indent=4)

if __name__ == "__main__":
    pass
