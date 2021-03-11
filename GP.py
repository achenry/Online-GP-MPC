import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pylab as plt
from matplotlib.transforms import Bbox
from mpl_settings import *
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
import pandas as pd
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
from scipy.spatial.distance import pdist, cdist, squareform


class GP:
    def __init__(self, device, input_labels, sampling_t_step, max_n_training_samples, n_init_training_samples,
                 n_test_samples, n_input_dims, n_output_dims, output_dim, prior_mean=0):

        self.device = device
        self.inputs = input_labels
        self.sampling_t_step = sampling_t_step
        self.sampling_clock = None
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.output_dim = output_dim
        self.max_n_training_samples = max_n_training_samples
        self.n_init_training_samples = n_init_training_samples
        self.n_available_training_samples = 0
        self.n_test_samples = n_test_samples
        self.test_indices = None
        self.x_test = None
        self.y_true = None
        self.training_indices = np.zeros((0, 1))

        self.x_train = np.zeros((0, self.n_input_dims))
        self.y_train = np.zeros((0, self.n_output_dims))

        self.init_x_train = np.zeros((0, self.n_input_dims))
        self.init_y_train = np.zeros((0, self.n_output_dims))

        self.online_x_train = np.zeros((0, self.n_input_dims))
        self.online_y_train = np.zeros((0, self.n_output_dims))

        self.inv_cov_train = [None for d in range(self.n_output_dims)]

        self.y_pred = np.zeros((0, self.n_output_dims))
        self.prior_mean = np.zeros((0, self.n_output_dims))
        self.y_std = np.zeros((0, self.n_output_dims))

        self.scores = [None for d in range(self.n_output_dims)]

        self.length_scales = None
        self.output_variances = None

        self.gp_fit = None
        self.meas_noises = None
        self.kernels = None
        self.interpolated_pred = None

        self.prior_mean = prior_mean

        self.prediction_figs = []
        self.prediction_axes = []

        self.score_fig = None
        self.score_ax = None

        self.prediction_fig_filename = None

    def mean_jacobian(self, x_test):
        # calculate derivative of posterior mean
        gradients = []
        for d in range(self.n_output_dims):
            if self.prior_mean:
                alpha = self.inv_cov_train[d] @ (self.y_train[:, d][:, np.newaxis] - np.mean(self.y_train[:, d]))
            else:
                alpha = self.inv_cov_train[d] @ (self.y_train[:, d][:, np.newaxis])

            cov_train_test = self.kernels[d](self.x_train, x_test)
            term_1 = -(np.linalg.inv(np.diag(self.length_scales[d] ** 2))) @ (x_test - self.x_train).T
            term_2 = cov_train_test * alpha
            gradients.append(term_1 @ term_2)

        return np.array(gradients).reshape((self.n_output_dims, self.n_input_dims))

    def covariance_jacobian(self, x_test):
        # TODO test
        # calculates derivative of covariance = kernel
        gradients = []
        for d in range(self.n_output_dims):
            cov_train_test = self.kernels[d](self.x_train, x_test)
            gradients.append(
                np.linalg.inv(np.diag(self.length_scales[d] ** 2)) @ (self.x_train - x_test).T @ cov_train_test)
        return np.array(gradients).reshape(-1, self.n_input_dims)

    def neg_log_marginal_likelihood(self, theta):

        n_training_samples, n_features = self.x_train.shape
        output_std = theta[0]
        length_scale = theta[1:1 + n_features]
        noise_variance = self.meas_noises[0]  # theta[-1]

        K = output_std ** 2 * RBF(length_scale=length_scale)(self.x_train) + noise_variance * np.eye(n_training_samples)

        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return np.inf

        alpha = cho_solve((L, True), self.y_train)

        data_fit = -0.5 * np.einsum("ik,ik->k", self.y_train, alpha)
        # data_fit = (-0.5 * self.y_train.T @ K_inv @ self.y_train).squeeze()
        # complexity_penalty = 0.5 * np.log(np.linalg.det(K_inv))
        complexity_penalty = np.log(np.diag(L)).sum()
        # K_inv = np.linalg.inv(K)
        # norm_const = 0.5 * n_training_samples * np.log(2 * np.pi)
        norm_const = 0.5 * n_training_samples * np.log(2 * np.pi)

        return -(data_fit - complexity_penalty - norm_const).sum(-1)

    def neg_log_marginal_likelihood_jacobian(self, theta):

        n_training_samples, n_features = self.x_train.shape
        output_std = theta[0]
        length_scale = theta[1:1 + n_features]
        noise_variance = self.meas_noises[0]  # theta[-1]

        K_unscaled = RBF(length_scale=length_scale)(self.x_train)
        K = output_std ** 2 * K_unscaled + noise_variance * np.eye(n_training_samples)

        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return np.zeros_like(theta)

        alpha = cho_solve((L, True), self.y_train)

        temp = np.einsum("ik,jk->ijk", alpha, alpha) \
               - cho_solve((L, True), np.eye(n_training_samples))[:, :, np.newaxis]

        dists = pdist(self.x_train / length_scale, metric='sqeuclidean')
        K = np.exp(-0.5 * dists)
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)

        dK_dov = 2 * output_std * K_unscaled[..., np.newaxis]
        dK_dl = (self.x_train[:, np.newaxis, :] - self.x_train[np.newaxis, :, :]) ** 2 \
                / (length_scale ** 3) * K[..., np.newaxis]
        # dK_dmn = np.eye(n_training_samples)[..., np.newaxis]

        grad = np.hstack([0.5 * np.einsum("ijl,jik->kl", temp, dK).sum(-1) for dK in [dK_dov, dK_dl]])

        return -grad

        # dK_dov = K_unscaled[..., np.newaxis]
        # dK_dl = (self.x_train[:, np.newaxis, :] - self.x_train[np.newaxis, :, :]) ** 2 \
        #              / (length_scale ** 2)
        # dK_dl *= K[..., np.newaxis]
        #
        # K_gradient = np.concatenate([dK_dov, dK_dl], axis=2)
        # dK_dn = np.eye(n_training_samples)
        #
        # cov_gradient = [dK_dov, dK_dn, dK_dl]

        # return -np.array([0.5 * np.trace(((alpha @ alpha.T) - K_inv) @ K_gradient[:, :, d]) for d in range(len(theta))])

    def set_kernel(self):
        pass

    def calculate_opt_hyperparams(self):
        # n_features = len(self.length_scales[0])
        # n_params = 1 + n_features
        # x0 = [self.output_variances[0]**0.5] + list(self.length_scales[0])# + self.meas_noises
        # # x0 = [1 for i in range(n_params)] + [1e-6]
        # # L-BFGS-B
        # jac = self.neg_log_marginal_likelihood_jacobian
        # # jac = None
        # # x0 = 0.1 * np.ones_like(x0)
        # res_opt = minimize(self.neg_log_marginal_likelihood,
        #                    x0=x0,
        #                    method='L-BFGS-B', jac=jac,
        #                    options={'maxiter': 15000, 'ftol': 1e-10})
        # theta = res_opt.x
        # self.output_variances[0] = theta[0] ** 2
        # self.length_scales[0] = theta[1:1 + n_features]

        # self.length_scales = np.array([[1.961970139642346, 7.728661224895884, 7.001233790723497]])
        # self.output_variance = np.array([15.630980809394867])
        # theta = [0, self.output_variances[0]**0.5] + list(self.length_scales[0])
        # # self.meas_noises[0] = theta[-1]
        # # self.kernels = [self.output_variances[0] * RBF(length_scale=self.length_scales[0])]
        # self.kernels = [ConstantKernel(constant_value=) + RBF(length_scale=self.length_scales[0])]
        # return theta
        return
        # n_training_samples, n_features = self.x_train.shape
        # # kernel = self.output_variances[0] * RBF(length_scale=self.length_scales[0])
        # gp = GaussianProcessRegressor(kernel=self.kernels[0], alpha=self.meas_noises[0], n_restarts_optimizer=25)
        # gp_fit = gp.fit(self.x_train, self.y_train)
        # # const = gp_fit.kernel_.theta[0] ** 2
        # self.output_variances[0] = gp_fit.kernel_.theta[0] ** 2
        # self.length_scales[0] = gp_fit.kernel_.theta[1:1 + n_features]
        # self.set_kernel()
        # self.kernels[0] = self.output_variances[0] * RBF(length_scale=self.length_scales[0]) \
        #                 + ConstantKernel(constant_value=const)
        # return gp_fit.kernel_.theta

    def update_inv_cov_train(self):

        for d in range(self.n_output_dims):
            cov_train = self.kernels[d](self.x_train)
            if self.meas_noises[d]:
                cov_train[np.diag_indices_from(cov_train)] += self.meas_noises[d]

            self.inv_cov_train[d] = np.linalg.inv(cov_train)

    def sq_exp_cov(self, x, y, output_variance, length_scale):
        return output_variance * np.exp(-(np.subtract.outer(x, y) ** 2) / (2 * length_scale ** 2))

    def score(self, y_true, y_pred):

        # for d in range(self.n_output_dims):
        res_sum_of_squares = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
        total_sum_of_squares = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0, dtype=np.float64)

        nonzero_num = res_sum_of_squares != 0
        nonzero_den = total_sum_of_squares != 0
        valid_score = nonzero_num & nonzero_den
        output_scores = np.ones([self.n_output_dims])
        output_scores[valid_score] = 1 - (res_sum_of_squares[valid_score] / total_sum_of_squares[valid_score])
        output_scores[nonzero_num & ~nonzero_den] = 0

        self.scores = output_scores
        return self.scores

    def predict(self, x_test):
        pred = []
        std = []
        if x_test.ndim == 1:
            x_test = x_test[np.newaxis, :]
        for d in range(self.n_output_dims):
            cov_test_train = self.kernels[d](x_test, self.x_train)
            cov_test = self.kernels[d](x_test, x_test)

            cov_inv = self.inv_cov_train[d]

            if self.prior_mean:
                mean = (self.prior_mean + cov_test_train @ cov_inv @
                        (self.y_train[:, d] - np.mean(self.y_train[:, d])))[:, np.newaxis]
            else:
                mean = (cov_test_train @ cov_inv @ self.y_train[:, d])[:, np.newaxis]

            pred.append(mean)

            cov = cov_test - (cov_test_train @ cov_inv @ cov_test_train.T)
            std.append(np.sqrt(np.diag(cov))[:, np.newaxis])

        pred = np.hstack(pred)
        std = np.hstack(std)

        return pred, std

    def set_training_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def collect_training_data(self, df, n_samples=1, is_init=False):

        # fetch new samples from db
        n_samples = min([n_samples, len(df.index)]) if type(n_samples) is int else len(df.index)

        if n_samples:

            df_spec = df.loc[df.is_init == is_init]

            new_samples = df_spec.head(n_samples)  # get(doc_id=len(next_state_table))

            # drop fetched samples from training data db
            df.drop(new_samples.index, inplace=True)  # remove(doc_ids=[len(next_state_table)])

            # add new samples to training dataset
            training_indices = np.vstack(new_samples['training_indices'].values)
            x_train = np.vstack(new_samples['x_train'].values)
            y_train = np.vstack(new_samples['y_train'].values)  # [:, self.output_dim][:, np.newaxis]

            # collect moving window of training samples
            self.training_indices = np.vstack([self.training_indices, training_indices])

            if is_init:
                self.init_x_train = np.vstack([self.init_x_train, x_train])
                self.init_y_train = np.vstack([self.init_y_train, y_train])
            else:
                self.online_x_train = np.vstack([self.online_x_train, x_train])
                self.online_y_train = np.vstack([self.online_y_train, y_train])

                max_online_n_training_samples = self.max_n_training_samples - self.n_init_training_samples
                n_available_online_training_samples = self.online_x_train.shape[0]
                self.online_x_train = self.online_x_train[
                               -min(max_online_n_training_samples, n_available_online_training_samples):, :]
                self.online_y_train = self.online_y_train[
                               -min(max_online_n_training_samples, n_available_online_training_samples):, :]

            self.x_train = np.vstack([self.init_x_train, self.online_x_train])
            self.y_train = np.vstack([self.init_y_train, self.online_y_train])

            # self.x_train = np.vstack([self.x_train, x_train])
            # n_available_training_samples = self.x_train.shape[0]

            # self.x_train = self.x_train[
            #                -min(self.max_n_training_samples, n_available_training_samples):, :]
            # self.y_train = np.vstack([self.y_train, y_train])[
            #                -min(self.max_n_training_samples, n_available_training_samples):, :]

            self.n_available_training_samples = self.x_train.shape[0]

        return df

    def collect_training_data_thread_func(self, df, is_simulation_running, is_gp_training, sampling_period):
        is_simulation_running.wait()
        is_gp_training.clear()
        print(f'\nCollecting Training Data for {self}\n')
        self.collect_training_data(df, min(1, len(df.index)))
        is_gp_training.set()
        print(f'\nDone Collecting Training Data for {self}\n')
        sleep(sampling_period)

    def plot_score(self, gp_df, label):
        fig, ax = plt.subplots(1, 1, frameon=False)
        # fig.suptitle(f'{label} GP Approximation Score $= 1 - \\frac{{RSS}}{{TSS}}$')
        ax.plot(gp_df['No. Training Samples'].values, gp_df['Score'].values)
        ax.set_xlabel('$N_{tr}$')
        ax.set_ylabel('Score', rotation=0)
        fig.show()
        return fig, ax

    def plot(self, y_pred, y_std, input_dims, input_labels, output_labels, plot_independent_data, show_fig=False):

        figs = []
        axes = []

        if self.x_train.shape[1] == 2 and not plot_independent_data:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            figs.append(fig)
            axes.append(ax)
            # fig.suptitle(f'{output_labels[self.output_dim]}')

            ax.set_xlabel(input_labels[0], labelpad=40)
            ax.set_ylabel(input_labels[1], labelpad=40, rotation=0)
            ax.set_zlabel('', labelpad=40)
            ax.tick_params(pad=20)

            # plot training data
            ax.scatter(self.x_train[:, input_dims[0]], self.x_train[:, input_dims[1]],
                       self.y_train[:, self.output_dim], label='Training Data', color='tab:blue', s=2.)

            # plot gp prediction
            ax.plot_trisurf(self.x_test[:, input_dims[0]], self.x_test[:, input_dims[1]], y_pred[:, self.output_dim],
                            color='tab:red',
                            label='GP Posterior Mean', antialiased=True, alpha=0.5, linewidth=0.5)

            # plot true value
            ax.plot_trisurf(self.x_test[:, input_dims[0]], self.x_test[:, input_dims[1]],
                            self.y_true[:, self.output_dim], label='True Function', antialiased=True, alpha=0.5)

            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)
            ax.set_zlim(auto=True)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # if show_fig:
            #     fig.show()
        else:

            n_training_samples, n_features = self.x_train.shape

            if plot_independent_data:
                for input_dim, label in zip(input_dims, input_labels):

                    fig, ax = plt.subplots(1, 1, frameon=False)
                    figs.append(fig)
                    axes.append(ax)
                    # fig.suptitle(output_labels[self.output_dim])

                    # plot based on first dim of y la only
                    ax.set_xlabel(label)
                    ax.set_ylabel(output_labels[self.output_dim], rotation=0)

                    n_training_samples_per_dim = int(self.max_n_training_samples / n_features)
                    X_train = self.x_train[
                              input_dim * n_training_samples_per_dim:(input_dim + 1) * n_training_samples_per_dim,
                              input_dim] \
                        if plot_independent_data else self.x_train[:, input_dim]
                    Y_train = self.y_train[
                              input_dim * n_training_samples_per_dim:(input_dim + 1) * n_training_samples_per_dim,
                              self.output_dim] \
                        if plot_independent_data else self.y_train[:, self.output_dim]
                    X_test = self.x_test[input_dim * self.n_test_samples:(input_dim + 1) * self.n_test_samples,
                             input_dim] \
                        if plot_independent_data else self.x_test[:, input_dim]
                    Y_pred = y_pred[input_dim * self.n_test_samples:(input_dim + 1) * self.n_test_samples,
                             self.output_dim] \
                        if plot_independent_data else y_pred[:, self.output_dim]
                    Y_std = y_std[input_dim * self.n_test_samples:(input_dim + 1) * self.n_test_samples,
                            self.output_dim] \
                        if plot_independent_data else y_std[:, self.output_dim]
                    Y_true = self.y_true[input_dim * self.n_test_samples:(input_dim + 1) * self.n_test_samples,
                             self.output_dim] \
                        if plot_independent_data else self.y_true[:, self.output_dim]

                    sort_idx = np.argsort(X_test, axis=0)
                    Y_pred = Y_pred[sort_idx]
                    Y_std = Y_std[sort_idx]
                    X_test = X_test[sort_idx]
                    Y_true = Y_true[sort_idx]

                    ax.plot(X_test, Y_true, color='tab:blue', label='True Function')

                    # plot gp prediction and variance

                    ax.plot(X_test, Y_pred, color='tab:red', label='GP Posterior Mean')
                    ax.fill_between(X_test, (Y_pred - 1 * Y_std), (Y_pred + 1 * Y_std), color='orangered',
                                    # alpha=0.5,#color='darkred',
                                    label='$\pm1*\sigma$')
                    ax.fill_between(X_test, (Y_pred - 2 * Y_std), (Y_pred + 2 * Y_std), color='coral',
                                    # alpha=0.25,#color='red',
                                    label='$\pm2*\sigma$')
                    ax.fill_between(X_test, (Y_pred - 3 * Y_std), (Y_pred + 3 * Y_std), color='lightcoral',
                                    # alpha=0.125,#color='coral',
                                    label='$\pm3*\sigma$')

                    # plot training data
                    ax.scatter(X_train, Y_train, color='tab:blue', label='Training Data')

                    ax.legend(bbox_to_anchor=(1, 1))
                    # ax.set_position([0, 0, 1, 1])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    # if show_fig:
                    #     fig.show()

            else:
                fig, ax = plt.subplots(2, 1)
                figs.append(fig)
                axes.append(ax)
                # fig.suptitle(output_labels[self.output_dim])
                ax[1].set_xlabel('time')
                ax[0].set_ylabel(output_labels[self.output_dim], rotation=0)

                ax[0].scatter(self.training_indices.astype('int') + 1, self.y_train[:, self.output_dim],
                              color='tab:blue', label='Training Data')
                ax[0].set_xlim(left=int(self.training_indices[0]) + 1, right=int(self.training_indices[-1]) + 1)

                ax[1].plot(self.test_indices.astype('int'), self.y_true, color='tab:blue', label='True Function')
                ax[1].set_xlim(left=int(self.test_indices[0]) + 1, right=int(self.test_indices[-1]) + 1)

                # plot gp prediction and variance
                Y_pred = y_pred[:, self.output_dim]
                Y_std = y_std[:, self.output_dim]
                ax[1].plot(self.test_indices, Y_pred, color='tab:red', label='GP Posterior Mean')

                ax[1].fill_between(self.test_indices.astype('int') + 1, (Y_pred - 1 * Y_std), (Y_pred + 1 * Y_std),
                                   color='orangered', label='$\pm1*\sigma$')
                ax[1].fill_between(self.test_indices.astype('int') + 1, (Y_pred - 2 * Y_std), (Y_pred + 2 * Y_std),
                                   color='coral', label='$\pm2*\sigma$')
                ax[1].fill_between(self.test_indices.astype('int') + 1, (Y_pred - 3 * Y_std), (Y_pred + 3 * Y_std),
                                   color='lightcoral', label='$\pm3*\sigma$')

                for a in ax:
                    a.spines['top'].set_visible(False)
                    a.spines['right'].set_visible(False)
                    a.legend(bbox_to_anchor=(1, 1))

                # if show_fig:
                # fig.show()
                #
                # ax[0].set_position(ax[0].get_position())
                # ax[1].set_position(ax[0].get_position())

        return figs, axes

    def generate_points_df(self, x_test):
        y_pred, y_std = self.predict(x_test)

        dict = {
            'x_train': list(self.x_train),
            'y_train': list(self.y_train),
            'x_test': list(x_test),
            'y_pred': list(y_pred),
            'y_std': list(y_std)
        }
        df = pd.DataFrame({key: pd.Series(val) for key, val in dict.items()})

        return df

    def test(self, data_reader, device, func, n_test_samples, n_simulation_steps, max_n_training_samples, mpc_t_step,
             results_dir, simulation_dir, simulation_name, run_mpc, use_linear_test_values, device_idx,
             output_dim=None):

        if func['use_gp']:
            is_state_gp = func['function_type'] == 'state'
            plot_independent_data = func['synthetic_data'] and not run_mpc

            # if using synthetically generated data from a closed form expression, generate random test samples
            if func['synthetic_data']:
                if self.x_test is None:
                    current_state_test, current_input_test, current_disturbance_test = \
                        data_reader.generate_current_data(device, n_test_samples, use_linear_test_values)

                if is_state_gp:
                    if self.x_test is None:
                        # generate lagged test input data
                        lagged_state_test, lagged_input_test, lagged_disturbance_test = \
                            data_reader.generate_lagged_data(n_test_samples, current_state_test,
                                                             current_input_test, current_disturbance_test)
                        # generate test inputs such that for each input row, only one element is nonzero,
                        # st the gp can be plotted for each input independent of the others
                        self.x_test = np.hstack([lagged_state_test, lagged_input_test, lagged_disturbance_test])

                    true_func = device.true_next_state_func
                    prior_func = device.next_state_prior_func
                else:
                    if self.x_test is None:
                        self.x_test = np.hstack([current_state_test, current_input_test, current_disturbance_test])

                    true_func = device.true_stage_cost_func
                    prior_func = device.stage_cost_prior_func

                n_test_samples, n_x_test_vars = self.x_test.shape

                if plot_independent_data:
                    x_test_ind = np.zeros((n_x_test_vars * n_test_samples, n_x_test_vars))
                    for c in range(n_x_test_vars):
                        x_test_ind[c * n_test_samples:(c + 1) * n_test_samples, c] = self.x_test[:, c]
                    self.x_test = x_test_ind

                if self.y_true is None:
                    y_true = []
                    # get the true next state error
                    n_test_samples = self.x_test.shape[0]
                    for n in range(n_test_samples):
                        device.set_simulation_step(int(n * func['sampling_t_step'] / mpc_t_step) % n_simulation_steps)
                        y_true.append(true_func(self.x_test[n]) - prior_func(self.x_test[n]))
                    self.y_true = np.vstack(y_true)[:, output_dim][:, np.newaxis]

            # else:
            #     self.n_test_samples = self.x_train.shape[0]
            #     self.x_test = self.x_train
            #     self.y_true = self.y_train

            # PLOT GP APPROXIMATION
            input_dims = [(func['state_cols'] + func['input_cols'] + func['disturbance_cols']).index(l)
                          for l in func['input_labels']]
            temp_input_labels = func['input_labels']

            input_labels = [f'{l}, $z_{{k, {i}}}$' for i, l in zip(input_dims, temp_input_labels)]

            if is_state_gp:

                output_dims = [func['output_state_col'].index(l) for l in func['output_labels']]
                # temp_output_labels = func['output_labels']
                # for i in range(len(output_dims)):
                #     temp_output_labels[output_dims[i]] = func['output_labels'][i]
                #
                # output_labels = [f'{device.name} {device.idx} State Variation, {l}, $f(z_{{k}})$' for i, l in
                #                  zip(output_dims, temp_output_labels)]
                output_labels = [f'$g_{o}^d(\mathbf{{z_k^d}})$' for o in output_dims]

            else:
                # output_labels = [f'{device.name} Stage Cost, $\overline{{l}}(z_k)$']
                output_labels = [f'$j^d(\mathbf{{z_k^d}})$']

            # title = 'State Variation' if is_state_gp else 'Stage Cost'
            points_filename = f'state_var_({device.name} {device.idx}, {output_dim})_points' if is_state_gp \
                else f'stage_cost_{device.name} {device.idx}_points'
            self.prediction_fig_filename = f'state_var_({device.name} {device.idx}, {output_dim})_gp' if is_state_gp \
                else f'stage_cost_{device.name} {device_idx}_gp'

            # plot gp approximation function values at the test inputs
            # plot true function values at the test inputs

            # fetch the predicted error is using the next state error gp,
            # else the predicted next state if using the next state gp
            y_pred, y_std = self.predict(self.x_test)

            if not os.path.exists(f'./{simulation_dir}/results/gp_results.csv'):
                gp_results_df = pd.DataFrame(
                    columns=['Prediction Name', 'Function', 'No. Training Samples', 'Length Scale',
                             'Output Variance', 'Measurement Noise', 'Score'])
            else:
                gp_results_df = pd.read_csv(f'./{simulation_dir}/results/gp_results.csv', engine='python', index_col=0,
                                            header=0)
                gp_results_df['No. Training Samples'] = gp_results_df['No. Training Samples'].astype(int)
                gp_results_df['Score'] = gp_results_df['Score'].astype(float)
                # gp_results_df.drop(gp_results_df.loc[gp_results_df['Prediction Name']
                #                                      == f'{output_labels[0]} Ntr={max_n_training_samples}'].index,
                #                    inplace=True)

            self.prediction_figs, self.prediction_axes = self.plot(y_pred, y_std, input_dims, input_labels,
                                                                   output_labels,
                                                                   plot_independent_data=plot_independent_data)

            self.score(self.y_true, y_pred)
            points_df = self.generate_points_df(self.x_test)
            points_df.to_csv(f'{results_dir}/{points_filename}')

            if plot_independent_data:
                n_features = self.x_train.shape[1]
                n_training_samples_per_dim = int(max_n_training_samples / n_features)
            else:
                n_training_samples_per_dim = max_n_training_samples

            function_name = f'State Variation ({device.name}, {device.idx}, {output_dim})' \
                if is_state_gp else f'Stage Cost ({device.name}, {device.idx})'

            gp_results = {'Prediction Name': f'{function_name} Ntr={n_training_samples_per_dim}',
                          'Function': function_name,
                          'No. Training Samples': n_training_samples_per_dim,
                          'Length Scale': func['length_scale'],
                          'Output Variance': func['output_variance'],
                          'Measurement Noise': func['meas_noise'],
                          'Score': self.scores[0]}

            existing_row_indices = (gp_results_df['Prediction Name'] == gp_results['Prediction Name']) \
                                   & (gp_results_df['Function'] == gp_results['Function'])

            if existing_row_indices.any(axis=0):
                gp_results_df = gp_results_df.loc[~existing_row_indices]
                # gp_results_df.loc[existing_row_indices[existing_row_indices].index[0], key] = value

            gp_results_df = gp_results_df.append(gp_results, ignore_index=True)
            gp_results_df = gp_results_df.reset_index(drop=True)

            gp_results_df.sort_values('No. Training Samples', inplace=True)

            gp_results_df.to_csv(f'./{simulation_dir}/results/gp_results.csv')

            self.score_fig, self.score_ax = self.plot_score(gp_results_df.loc[gp_results_df['Function']
                                                                              == gp_results['Function']],
                                                            output_labels[0])

    def update_device_bounds(self, k0, n_horizon):
        orig_params = self.device.original_parameters
        params = self.device.parameters
        for i, input in enumerate(self.inputs):
            min_val = np.min(self.x_train[:, i])
            min_label = input[:-2] + '_min'
            max_label = input[:-2] + '_max'
            max_val = np.max(self.x_train[:, i])
            if min_label in params:
                # for j in range(len(params[min_label])):
                for j in range(k0, k0 + n_horizon):
                    # if orig_params[min_label][j] != 0:
                    params[min_label][j] = np.max([orig_params[min_label][j], min_val])

            if max_label in params:
                # for j in range(len(params[max_label])):
                for j in range(k0, k0 + n_horizon):
                    # if orig_params[max_label][j] != 0:
                    params[max_label][j] = np.min([orig_params[max_label][j], max_val])


class NextStateGP(GP):

    def __init__(self, device, dim, input_labels, state_output_variances, state_meas_noises, state_length_scales,
                 sampling_t_step,
                 max_n_training_samples, n_init_training_samples, n_test_samples,
                 state_lag, input_lag, disturbance_lag, output_lag,
                 n_states, n_inputs, n_disturbances, n_outputs, n_input_dims=None, is_true_system=False, prior_mean=0):
        self.dim = dim

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_disturbances = n_disturbances
        self.n_outputs = n_outputs

        self.state_lag = state_lag
        self.input_lag = input_lag
        self.disturbance_lag = disturbance_lag
        self.output_lag = output_lag

        self.n_input_dims = n_states * (self.state_lag + 1) \
                            + n_inputs * (self.input_lag + 1) \
                            + n_disturbances * (self.disturbance_lag + 1) if n_input_dims is None else n_input_dims

        super().__init__(device=device, input_labels=input_labels,
                         sampling_t_step=sampling_t_step,
                         max_n_training_samples=max_n_training_samples,
                         n_init_training_samples=n_init_training_samples,
                         n_test_samples=n_test_samples,
                         n_input_dims=self.n_input_dims,
                         n_output_dims=1,
                         output_dim=0,
                         prior_mean=prior_mean)

        self.output_variances = [state_output_variances]
        self.meas_noises = [state_meas_noises]
        self.length_scales = np.array([state_length_scales])

        self.kernels = None  # Matern(length_scale=2, nu=1.5)
        self.set_kernel()

    def set_kernel(self, const_add=0):
        self.kernels = [ConstantKernel(self.output_variances[0]) * RBF(length_scale=self.length_scales[0])]
                        # + ConstantKernel(constant_value=const_add)]
        # if add_const_kernel:
        #     for k in range(len(self.kernels)):
        #         self.kernels[k] = self.kernels[k] + ConstantKernel(constant_value=self.device.ref_state[self.output_dim])


class StageCostGP(GP):

    def __init__(self, device, input_labels, cost_output_variance, cost_meas_noise, cost_length_scale, sampling_t_step,
                 max_n_training_samples, n_init_training_samples, n_test_samples,
                 n_states, n_inputs, n_disturbances):
        super().__init__(device=device, input_labels=input_labels,
                         sampling_t_step=sampling_t_step,
                         max_n_training_samples=max_n_training_samples,
                         n_init_training_samples=n_init_training_samples,
                         n_test_samples=n_test_samples,
                         n_input_dims=n_states + n_inputs + n_disturbances,
                         n_output_dims=1,
                         output_dim=0)

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.output_variances = [cost_output_variance]
        self.meas_noises = [cost_meas_noise]
        self.length_scales = np.array([cost_length_scale])

        self.kernels = None
        self.set_kernel()

    def set_kernel(self, add_const_kernel=False):
        self.kernels = [ConstantKernel(self.output_variances[0]) * RBF(length_scale=self.length_scales[0])]
