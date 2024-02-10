import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import odeint
from tqdm import tqdm

from ._utils import sample_lhs, save_to_csv


class ODEDataset:
    def __init__(self, args, params_config):
        self.args = args
        self.params_config = params_config
        self.ode_dim = self.params_config["ode_dim"]
        self.ode_name = self.params_config["task"]
        self.params, self.y0 = self._get_ode_params_and_y0(self.args.num_env, self.args.params_strategy)
        self.t_series = None
        self.train_indices, self.test_indices = None, None
        self.num_train, self.num_test = None, None
        self.n = int(self.params_config["t_max"] / self.params_config["dt"])
        self.dt = self.params_config["dt"]
        self.y = np.zeros([self.args.num_env, self.n, self.ode_dim])
        self.y_noise = np.zeros([self.args.num_env, self.n, self.ode_dim])
        self.dy_noise = np.zeros([self.args.num_env, self.n, self.ode_dim])
        self.setup_seed(self.args.seed)
        self._set_t()
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
            random_rate=0.3,)
        y0 = params_func(
            num_env=num_env,
            default_list=self.params_config["default_y0_list"],
            base=self.params_config["random_y0_base"],
            seed=self.args.seed,
            random_rate=0.3, )
        return params, y0

    def _func(self, x, t, env_id):
        raise NotImplemented

    def _set_t(self):
        assert self.args.sample_strategy in ["uniform", "lhs"]
        if self.args.sample_strategy == "uniform":
            self.t_series = np.linspace(self.params_config["t_min"],
                                        self.params_config["t_max"] - self.params_config["dt"], self.n)
        else:  # lhs
            self.t_series = sample_lhs(self.params_config["t_min"], self.params_config["t_max"], self.n)
        # print(self.t_series)

    def build(self):
        train_test_total = self.args.train_test_total
        self.num_train = int(self.args.train_ratio * train_test_total)
        self.num_test = train_test_total - self.num_train
        indices = np.arange(self.n)
        assert self.num_train + self.num_test <= self.n
        if self.args.train_sample_strategy == "uniform":
            assert self.args.dataset_sparse in ["sparse", "dense"]
            if self.args.dataset_sparse == "sparse":
                train_indices = indices[0::(int(self.n / 1.01) // self.num_train)][:self.num_train]
                test_indices = indices[0::(int(self.n / 1.01) // self.num_test)][:self.num_test]
            else:
                sample_freq = int(0.03 / self.dt) if int(0.05 / self.dt) >= 1 else 1
                train_indices = indices[::sample_freq][:self.num_train]
                test_indices = indices[::sample_freq][:self.num_test]
        else:  # random
            np.random.shuffle(indices)
            train_indices = sorted(indices[:self.num_train])
            test_indices = sorted(indices[self.num_train: self.num_train + self.num_test])
        self.train_indices = train_indices
        self.test_indices = test_indices

        ## for SPL only
        # save_folder = os.path.join(self.args.save_folder, self.ode_name, self.args.time_string)
        save_folder = os.path.join(self.args.save_folder, self.ode_name, f"{self.args.noise_ratio:.3f}")

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        print(f"ode_name: {self.ode_name}")
        print(f"num_env: {self.args.num_env}")
        print(f"t_min: {self.params_config['t_min']}, t_max: {self.params_config['t_max']}, dt: {self.params_config['dt']}")
        print(f"train set: {self.num_train} ({self.num_train / train_test_total * 100:.1f} %), test set: {self.num_test} ({self.num_test / train_test_total * 100:.1f} %), train_test_total: {train_test_total}, n_total: {self.n}")
        print(f"noise ratio: {self.args.noise_ratio}")
        print(f"save_folder: {save_folder}")

        self._save_task_info(save_folder)

        for i in range(self.args.num_env):
            print(f"Environment {i:02d}: params={[f'{item:.8f}' for item in self.params[i]]}, y0={[f'{item:.8f}' for item in self.y0[i]]}, truth = {[item.format(*self.params[i]) for item in (self.params_config['truth_ode_format'] if self.params_config.get('truth_ode_format') else [])]}")


        for i in tqdm(range(self.args.num_env)):
            self.y[i] = odeint(self._func, self.y0[i], self.t_series, (i,))
            self.y_noise[i] = self.y[i] * (1 + 2.0 * self.args.noise_ratio * (-0.5 + np.random.random(self.y[i].shape)))

            y_noise_train, y_noise_test = self.y_noise[i][train_indices], self.y_noise[i][test_indices]
            raw_path = os.path.join(save_folder, f"{self.ode_name}_{i}_raw.csv")
            save_to_csv(raw_path, [self.t_series] + [self.y_noise[i][:, j] for j in range(self.ode_dim)], ["t"] + self.params_config["curve_names"][:self.ode_dim])

            if self.args.save_figure:
                t_train, t_test = self.t_series[train_indices], self.t_series[test_indices]
                y_train, y_test = self.y[i][train_indices], self.y[i][test_indices]

                save_path = os.path.join(save_folder, f"{self.ode_name}_{i}.png")
                self._plot_dataset(save_path, self.t_series, t_train, t_test, self.y[i], y_train, y_test, y_noise_train, y_noise_test)

            for j in range(self.ode_dim):
                self.dy_noise[i][:, j] = np.gradient(self.y_noise[i][:, j], self.t_series)
                # dy_noise_train = self.dy_noise[i][:, j][train_indices]
                # dy_noise_test = self.dy_noise[i][:, j][test_indices]
                # dy_path_train = os.path.join(save_folder, f"{self.ode_name}_d{self.params_config['curve_names'][j]}_train_{i}.csv")
                # dy_path_test = os.path.join(save_folder, f"{self.ode_name}_d{self.params_config['curve_names'][j]}_test_{i}.csv")
                # save_to_csv(dy_path_train, [y_noise_train[:, idx] for idx in range(self.ode_dim)] + [dy_noise_train])
                # save_to_csv(dy_path_test, [y_noise_test[:, idx] for idx in range(self.ode_dim)] + [dy_noise_test])
            dy_path_train = os.path.join(save_folder, f"{self.ode_name}_train_{i}.csv")
            dy_path_test = os.path.join(save_folder, f"{self.ode_name}_test_{i}.csv")
            save_to_csv(dy_path_train, 
                        [self.t_series[train_indices]] + [y_noise_train[:, j] for j in range(self.ode_dim)] + [self.dy_noise[i][train_indices, j] for j in range(self.ode_dim)], 
                        ["t"] + self.params_config["curve_names"][:self.ode_dim] + ['d'+name for name in self.params_config["curve_names"][:self.ode_dim]])
            save_to_csv(dy_path_test, 
                        [self.t_series[test_indices]] + [y_noise_test[:, j] for j in range(self.ode_dim)] + [self.dy_noise[i][test_indices, j] for j in range(self.ode_dim)], 
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
