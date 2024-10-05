import numpy as np
import torch
import gpytorch
from scipy.interpolate import UnivariateSpline
import tqdm
from numpy.lib.stride_tricks import sliding_window_view


def get_spline(times, data, window_size=10, coef=10):
    """
    Compute the time-weighted exponential moving average of a 1D numpy array
    using a sliding window approach.

    Parameters:
    - data (numpy array): The input data.
    - times (numpy array): The time points corresponding to the data values.
    - window_size (int): The size of the window to compute the weighted EMA.

    Returns:
    - numpy array: The time-weighted EMA of the input data with NaN at the ends.
    """    
    
    times = np.pad(times, (window_size//2,window_size//2), constant_values=(0,0))
    data = np.pad(data, (window_size//2,window_size//2), constant_values=(0,0))
    n = len(data)
    ema = np.full(n, np.nan)  # Initialize the EMA array with NaNs
    mask_notnans = ~np.isnan(data)
    
    # Create a sliding window view of the data and times
    data_windows = sliding_window_view(data[mask_notnans], window_shape=window_size)
    time_windows = sliding_window_view(times[mask_notnans], window_shape=window_size)
    
    # Compute the EMA for each window
    for i in range(data_windows.shape[0]):
        window_data = data_windows[i]
        window_times = time_windows[i]
        
       # Calculate weights based on time intervals within the window
        weights = np.exp(-coef*np.abs(window_times[window_size // 2 + 1] - window_times))
#         weights = np.exp(-(window_times[-1] - window_times))
        weights /= weights.sum()  # Normalize weights to sum to 1
        
        # Calculate the weighted average for the current window
        ema[np.where(mask_notnans)[0][i + window_size // 2]] = np.sum(weights * window_data)

    # Evaluate the spline fits on a dense grid of x values for plotting
    paddings = window_size//2
#     print(times[mask_notnans][paddings:-paddings], data[mask_notnans][paddings:-paddings])
    spline = UnivariateSpline(times[mask_notnans][paddings:-paddings], 
                              ema[mask_notnans][paddings:-paddings], 
                              s=0)
    ema[np.where(mask_notnans)[0][paddings:-paddings]] = spline(times[mask_notnans][paddings:-paddings])
    return spline

class SplineMeanFunction(gpytorch.means.Mean):
    def __init__(self, x, y):
        super().__init__()
        self.spline = get_spline(x.squeeze(), y.squeeze(), window_size=5, coef=1)  # Ensure x and y are 1D for spline

    def forward(self, x):
        x_np = x.detach().cpu().numpy().squeeze()  # Squeeze to ensure it's 1D for spline
        y_spline = self.spline(x_np)
        return torch.tensor(y_spline, dtype=x.dtype, device=x.device).squeeze()  # Squeeze output_v20240301 to 1D

class GPModelWithFixedNoise(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithFixedNoise, self).__init__(train_x, train_y, likelihood)
        self.mean_module = SplineMeanFunction(train_x, train_y)  # Your custom mean
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class GPPCA0:
    def __init__(self, Y, t, sigma, sigma_out=None, sigma_in=None, r=1, it=1000):
        self.Y = Y
        self.t = t
        self.sigma = sigma
        self.r = r
        self.n_traj = Y.shape[1]

        if sigma_out is None:
            self.sigma_out = np.std(Y)
        else:
            self.sigma_out = sigma_out

        if sigma_in is None:
            self.sigma_in = self.t[1] - self.t[0]
        else:
            self.sigma_in = sigma_in

#         self.K = self.get_kernel_discrete()
#         self.A = self.get_factor_loading()
        fixed_noise_value = sigma
        train_x, train_y = torch.tensor(t).view(-1,1), torch.tensor(Y).view(-1)
        training_iterations = 1000

        # Assuming train_x and train_y are your data tensors
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.tensor([fixed_noise_value]),  # Replace fixed_noise_value with your noise level
            learn_additional_noise=True
        )
        self.model = GPModelWithFixedNoise(train_x, train_y, self.likelihood)

        self.model.train()
        self.likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        def train():
            for i in tqdm.trange(training_iterations):  # Number of iterations
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                loss.sum().backward()
                optimizer.step()

        train()
        self.model.eval()
        self.likelihood.eval()

        # Assume t is given as a numpy array or a list of time points
        t_new = np.linspace(min(t), max(t), len(t)*5)  # Example time points

        # Conversion and prediction steps
        t_tensor = torch.tensor(t_new, dtype=torch.float32).view(-1,1)  # Reshape if necessary
        t_tensor = t_tensor.to(self.model.covar_module.base_kernel.lengthscale.device)  # Adjust for model's device

        self.model.eval()
        self.likelihood.eval()
        


    def get_predictive(self, new_sample=1, t_new=None):
        if new_sample == 1:
            return self.get_predictive_mean(1, t_new)
        # predictive distribution
        t_tensor = torch.tensor(t_new, dtype=torch.float32).view(-1,1)  # Reshape if necessary
        t_tensor = t_tensor.to(self.model.covar_module.base_kernel.lengthscale.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(t_tensor))
            samples = observed_pred.sample(torch.Size([new_sample]))
        return samples

    def get_predictive_mean(self, new_sample=1, t_new=None):
        assert new_sample == 1
        # predictive distribution
        t_tensor = torch.tensor(t_new, dtype=torch.float32).view(-1,1)  # Reshape if necessary
        t_tensor = t_tensor.to(self.model.covar_module.base_kernel.lengthscale.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(t_tensor))
            mean = observed_pred.mean
        return mean

    def get_X_cov(self, t_new=None):
        t_tensor = torch.tensor(t_new, dtype=torch.float32).view(-1,1)  # Reshape if necessary
        t_tensor = t_tensor.to(self.model.covar_module.base_kernel.lengthscale.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(t_tensor))
            covariance = observed_pred.covariance_matrix
        return covariance


# # Code from https://github.com/ZhaozhiQIAN/D-CODE-ICLR-2022/blob/main/gppca.py#L137
# import numpy as np
# from scipy.optimize import line_search, bracket
# from scipy.optimize import minimize

# def get_rbf_kernel(t, sigma_out, sigma_in, t2=None):
#     tc = t[:, None]
#     if t2 is None:
#         tr = t
#     else:
#         tr = t2
#     t_mat = sigma_out ** 2 * np.exp(-1. / (2 * sigma_in ** 2) * (tc - tr) ** 2)
#     return t_mat


# class GPPCA0:
#     def __init__(self, Y, t, sigma, sigma_out=None, sigma_in=None, r=1):
#         self.Y = Y
#         self.t = t
#         self.sigma = sigma
#         self.r = r
#         self.n_traj = Y.shape[1]

#         if sigma_out is None:
#             self.sigma_out = np.std(Y)
#         else:
#             self.sigma_out = sigma_out

#         if sigma_in is None:
#             self.sigma_in = self.t[1] - self.t[0]
#         else:
#             self.sigma_in = sigma_in

#         self.K = self.get_kernel_discrete()
#         self.A = self.get_factor_loading()

#     def get_hyper_param(self, method='Powell'):
#         x0 = np.log(np.array([self.sigma_in]))
#         res = minimize(self.loss_fn, x0=x0, method=method)
#         # print(res)
#         self.sigma_in = np.exp(res['x'][0])

#     def loss_fn(self, x):
#         # input in log scale
#         sigma_out = self.sigma_out
#         sigma_in = x[0]
#         sigma_in = np.exp(sigma_in)

#         tau = sigma_out ** 2 / self.sigma ** 2

#         K = get_rbf_kernel(self.t, sigma_out, sigma_in)

#         # T, T
#         W = np.linalg.inv(1. / tau * np.linalg.inv(K) + np.eye(K.shape[0]))
#         # R, T
#         b = np.matmul(self.Y, self.A).T

#         S = np.abs(np.sum(self.Y ** 2) - np.sum(np.diag(b @ W @ b.T)))

#         f2 = np.log(S) * (-1 * self.Y.shape[0] * self.Y.shape[1] / 2)

#         f1 = -1. / 2 * self.r * np.linalg.slogdet(tau * K + np.eye(K.shape[0]))[1]

#         return -1 * (f1 + f2)

#     def get_predictive(self, new_sample=1, t_new=None):
#         if new_sample == 1:
#             return self.get_predictive_mean(1, t_new)
#         # predictive distribution
#         Z_hat = self.get_factor(t_new=t_new)
#         X_hat = self.get_X_mean(Z_hat)
#         K_fac = self.get_X_cov(t_new=t_new)

#         X_list = list()
#         for i in range(new_sample):
#             noise = self.sample_noise(K_fac)
#             X_sample = X_hat + noise
#             X_sample = X_sample[:, None, :]
#             X_list.append(X_sample)

#         X = np.concatenate(X_list, axis=1)
#         return X

#     def get_predictive_mean(self, new_sample=1, t_new=None):
#         assert new_sample == 1
#         # predictive distribution
#         Z_hat = self.get_factor(t_new=t_new)
#         X_hat = self.get_X_mean(Z_hat)
#         return X_hat

#     def get_factor_loading(self):
#         G = self.get_G()
#         w, v = np.linalg.eigh(G)
#         A = v[:, -self.r:]
#         return A

#     def get_X_cov(self, t_new=None):
#         if t_new is None:
#             K = self.K
#         else:
#             K = get_rbf_kernel(t_new, self.sigma_out, self.sigma_in, t_new)

#         # T, D
#         D = self.sigma ** 2 * np.linalg.inv(1. / self.sigma ** 2 * np.linalg.inv(K) + np.eye(K.shape[0]))
#         try:
#             D_fac = np.linalg.cholesky(D)
#         except np.linalg.LinAlgError:
#             w, v = np.linalg.eigh(D)
#             w_pos = w
#             w_pos[w_pos < 0] = 0
#             w_pos = np.sqrt(w_pos)
#             D_fac = v @ np.diag(w_pos)

#         # B, R
#         A_fac = self.A
#         K_fac = np.kron(D_fac, A_fac)
#         return K_fac

#     def sample_noise(self, K_fac):
#         noise = np.random.randn(K_fac.shape[1])
#         vec = K_fac @ noise
#         mat = vec.reshape(len(vec) // self.n_traj, self.n_traj)
#         return mat

#     def get_factor(self, t_new=None):
#         if t_new is None:
#             f1 = self.K
#         else:
#             f1 = get_rbf_kernel(t_new, self.sigma_out, self.sigma_in, self.t)

#         # print(f1.shape)
#         # print(self.K.shape)
#         Z_hat = f1 @ np.linalg.inv(self.K + self.sigma ** 2 * np.eye(self.K.shape[0])) @ self.Y @ self.A
#         return Z_hat

#     def get_X_mean(self, Z_hat):
#         X_hat = np.matmul(Z_hat, self.A.T)
#         return X_hat

#     def get_kernel_discrete(self):
#         sigma_out = self.sigma_out
#         sigma_in = self.sigma_in
#         K = get_rbf_kernel(self.t, sigma_out, sigma_in)
#         return K

#     def get_G(self):
#         # get G matrix - Thm 3 Eq 7

#         W = np.linalg.inv(self.sigma ** 2 * np.linalg.inv(self.K) + np.eye(self.K.shape[0]))
#         G = np.matmul(np.matmul(self.Y.transpose(), W), self.Y)
#         return G