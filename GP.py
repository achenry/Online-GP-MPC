import numpy as np
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pylab as plt
from mpl_settings import *
# from mpl_toolkits.mplot3d import Axes3D
from time import sleep
import pandas as pd
import os


class GP:
    def __init__(self, n_training_samples, n_test_samples, n_input_dims, n_output_dims, output_dim):

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.output_dim = output_dim
        self.n_training_samples = n_training_samples
        self.n_test_samples = n_test_samples
        self.x_test = np.zeros((0, self.n_input_dims))
        self.y_test = np.zeros((0, self.n_output_dims))
        self.x_train = np.zeros((0, self.n_input_dims))
        self.y_train = np.zeros((0, self.n_output_dims))
        self.inv_cov_train = [None for d in range(self.n_output_dims)]

        self.y_pred = np.zeros((0, self.n_output_dims))
        self.y_std = np.zeros((0, self.n_output_dims))

        self.scores = [None for d in range(self.n_output_dims)]

        self.length_scales = None
        self.output_variances = None

        self.gp_fit = None
        self.meas_noises = None
        self.kernels = None
        self.interpolated_pred = None

    def mean_jacobian(self, x_test):
        # calculate derivative of posterior mean
        gradients = []
        for d in range(self.n_output_dims):
            alpha = self.inv_cov_train[d] @ self.y_train[:, d]
            cov_train_test = self.kernels[d](self.x_train, [x_test])
            gradients.append(-((1 / self.length_scales[d] ** 2) * (x_test - self.x_train)).T
                             @ np.array([cov_train_test[n] * alpha[n] for n in range(self.n_training_samples)]))

        return np.array(gradients).reshape((self.n_output_dims, self.n_input_dims))

    def covariance_jacobian(self, x_test):
        # TODO test
        # calculates derivative of covariance = kernel
        gradients = []
        for d in range(self.n_output_dims):
            cov_train_test = self.kernels[d](self.x_train, [x_test])
            gradients.append((1 / self.length_scales[d] ** 2) @ np.array([cov_train_test[s, :] * (self.x_train[s, :] - x_test)
                                         for s in range(self.n_training_samples)]))

    def log_marginal_likelihood(self, theta):
        output_variance = theta[0]
        length_scale = theta[1]
        noise_variance = theta[2]
        n_training_samples = self.x_train.shape[0]
        cov_train = output_variance * RBF(length_scale=length_scale)(self.x_train, self.x_train) + noise_variance \
                    * np.eye(n_training_samples)

        inv_cov_train = np.inv(cov_train)
        data_fit = -0.5 * self.y_train.T @ inv_cov_train @ self.y_train
        complexity_penalty = 0.5 * np.log(np.abs(inv_cov_train))
        norm_const = 0.5 * n_training_samples * np.log(2 * np.pi)

        return data_fit - complexity_penalty - norm_const

    def log_marginal_likelihood_gradient(self, theta):

        output_variance = theta[0]
        length_scale = theta[1]
        noise_variance = theta[2]
        n_training_samples = self.x_train.shape[0]

        alpha = self.inv_cov_train @ self.y_train

        dK_dov = lambda x, y: RBF(length_scale=length_scale)(x, y)
        dK_dl = lambda x, y: (output_variance / length_scale) @\
                             RBF(length_scale=length_scale)(x, y) @ (np.subtract.outer(x, y) ** 2)
        dK_dn = lambda x, y: np.eye(n_training_samples)

        cov_gradient = [dK_dov, dK_dl, dK_dn]

        return 0.5 * np.trace((alpha @ alpha.T - self.inv_cov_train) @ cov_gradient)

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

        for d in range(self.n_output_dims):

            # TODO all zero values for TCL next_state because values are too dissimilar
            cov_test_train = self.kernels[d](x_test, self.x_train)
            cov_test = self.kernels[d](x_test, x_test)

            cov_inv = self.inv_cov_train[d]

            pred.append((cov_test_train @ cov_inv @ self.y_train[:, d])[:, np.newaxis])
            cov = cov_test - (cov_test_train @ cov_inv @ cov_test_train.T)

            std.append(np.sqrt(np.diag(cov))[:, np.newaxis])

        pred = np.hstack(pred)
        std = np.hstack(std)

        return pred, std

    def set_training_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def collect_training_data(self, df, n_samples=1):

        # fetch new samples from db
        n_samples = min([n_samples, len(df.index)]) if type(n_samples) is int else len(df.index)

        if n_samples:
            new_samples = df.tail(n_samples)  # get(doc_id=len(next_state_table))

            # drop fetched samples from training data db
            df.drop(new_samples.index, inplace=True)  # remove(doc_ids=[len(next_state_table)])

            # add new samples to training dataset
            x_train = np.vstack(new_samples['x_train'].values)
            y_train = np.vstack(new_samples['y_train'].values)  # [:, self.output_dim][:, np.newaxis]

            # collect moving window of training samples
            self.x_train = np.vstack([self.x_train, x_train])
            n_training_samples = self.x_train.shape[0]
            self.x_train = self.x_train[-min(self.n_training_samples, n_training_samples):, :]
            self.y_train = np.vstack([self.y_train, y_train])[-min(self.n_training_samples, n_training_samples):, :]
            self.n_training_samples = self.y_train.shape[0]

            self.update_inv_cov_train()

    def collect_training_data_thread_func(self, df, is_simulation_running, sampling_period):
        while True: #not is_simulation_running.is_set():
            self.collect_training_data(df, min(1, len(df.index)))
            sleep(sampling_period)

    def plot_score(self, gp_df):
        fig, ax = plt.subplots(1, 1, frameon=False)
        fig.suptitle(f'GP Approximation Score $= 1 - \\frac{{RSS}}{{TSS}}$')
        ax.plot(gp_df['No. Training Samples'].values, gp_df['Score'].values)
        fig.show()
        return fig, ax

    def plot(self, x_test, y_true, y_pred, y_std,
             input_dims, input_labels, output_labels, title, independent_data):

        figs = []
        axes = []
        for d in range(len(input_labels)):

            # plot based on first dim of y la only
            fig, ax = plt.subplots(1, 1, frameon=False)
            figs.append(fig)
            axes.append(ax)
            fig.suptitle(title)

            ax.set_xlabel(input_labels[d])
            ax.set_ylabel(output_labels[self.output_dim], rotation=0)

            X_train = self.x_train[d * self.n_training_samples:(d + 1) * self.n_training_samples, input_dims[d]] \
                    if independent_data else self.x_train[:, input_dims[d]]
            Y_train = self.y_train[d * self.n_training_samples:(d + 1) * self.n_training_samples, self.output_dim] \
                    if independent_data else self.y_train[:, self.output_dim]
            X_test = x_test[d * self.n_test_samples:(d + 1) * self.n_test_samples, input_dims[d]] \
                    if independent_data else x_test[:, input_dims[d]]
            Y_pred = y_pred[d * self.n_test_samples:(d + 1) * self.n_test_samples, self.output_dim] \
                    if independent_data else y_pred[:, self.output_dim]
            Y_std = y_std[d * self.n_test_samples:(d + 1) * self.n_test_samples, self.output_dim] \
                    if independent_data else y_std[:, self.output_dim]
            Y_true = y_true[d * self.n_test_samples:(d + 1) * self.n_test_samples, self.output_dim] \
                if independent_data else y_true[:, self.output_dim]

            sort_idx = np.argsort(X_test, axis=0)
            Y_pred = Y_pred[sort_idx]
            Y_std = Y_std[sort_idx]
            X_test = X_test[sort_idx]
            Y_true = Y_true[sort_idx]

            ax.plot(X_test, Y_true, label='True Function')

            # plot gp prediction and variance

            ax.plot(X_test, Y_pred, 'red', label='GP Posterior Mean')
            ax.fill_between(X_test, (Y_pred - 3 * Y_std), (Y_pred + 3 * Y_std), color='coral', label='$\pm3*\sigma$')
            ax.fill_between(X_test, (Y_pred - 2 * Y_std), (Y_pred + 2 * Y_std), color='red', label='$\pm2*\sigma$')
            ax.fill_between(X_test, (Y_pred - 1 * Y_std), (Y_pred + 1 * Y_std), color='darkred', label='$\pm1*\sigma$')

            # plot training data
            ax.scatter(X_train, Y_train, label='Training Data')

            ax.legend(bbox_to_anchor=(1, 1))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.show()

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

    def test(self, data_reader, device, func, n_test_samples, n_simulation_steps, n_training_samples, mpc_t_step,
             results_dir, simulation_dir, simulation_name, run_mpc, use_linear_test_values, device_idx, state_idx=None):

        if func['use_gp']:
            is_state_gp = func['function_type'] == 'state'
            independent_data = func['training_data_path'] is None

            # if using synthetically generated data from a closed form expression, generate random test samples
            if func['synthetic_data']:
                current_state_test, current_input_test, current_disturbance_test = \
                    data_reader.generate_current_data(device, n_test_samples, use_linear_test_values)

                if is_state_gp:
                    # generate lagged test input data
                    lagged_state_test, lagged_input_test, lagged_disturbance_test = \
                        data_reader.generate_lagged_data(n_test_samples, current_state_test,
                                                         current_input_test, current_disturbance_test)
                    # generate test inputs such that for each input row, only one element is nonzero,
                    # st the gp can be plotted for each input independent of the others
                    x_test = np.hstack([lagged_state_test, lagged_input_test, lagged_disturbance_test])

                    y_true = []
                    # get the true next state error
                    for n in range(x_test.shape[0]):
                        device.set_simulation_step(int(n * func['sampling_t_step'] / mpc_t_step) % n_simulation_steps)
                        y_true.append(device.true_next_state_func(x_test[n]) - device.next_state_prior_func(x_test[n]))
                    y_true = np.vstack(y_true)

                else:
                    x_test = np.hstack([current_state_test, current_input_test, current_disturbance_test])
                    y_true = []
                    for n in range(n_test_samples):
                        device.set_simulation_step(int(n * func['sampling_t_step'] / mpc_t_step) % n_simulation_steps)
                        y_true.append([device.true_stage_cost_func(x_test[n]) - device.stage_cost_prior_func(x_test[n])])
                    y_true = np.vstack(y_true)

                # independent_data = True
                if independent_data:
                    n_x_test_vars = x_test.shape[1]
                    x_test_ind = np.zeros((n_x_test_vars * n_test_samples, n_x_test_vars))
                    for c in range(n_x_test_vars):
                        x_test_ind[c * n_test_samples:(c + 1) * n_test_samples, c] = x_test[:, c]
                    x_test = x_test_ind
            else:
                self.n_test_samples = self.x_train.shape[0]
                x_test = self.x_train
                y_true = self.y_train

            # PLOT GP APPROXIMATION
            if is_state_gp:
                input_dims = [(device.states + device.inputs
                               + device.disturbances).index(l) for l in func['input_labels']]

                temp_input_labels = func['input_labels']

                input_labels = [f'{l}, $z_{{k, {i}}}$' for i, l in zip(input_dims, temp_input_labels)]

                output_dims = [device.states.index(l) for l in func['output_labels']]
                temp_output_labels = func['output_labels']
                for i in range(len(output_dims)):
                    temp_output_labels[output_dims[i]] = func['output_labels'][i]

                output_labels = [f'{l}, $x_{{k + 1, {i}}}$' for i, l in zip(output_dims, temp_output_labels)]

            else:
                input_dims = [(device.states + device.inputs + device.disturbances).index(l)
                              for l in func['input_labels']]
                temp_input_labels = func['input_labels']

                input_labels = [f'{l}, $z_{{k, {i}}}$' for i, l in zip(input_dims, temp_input_labels)]

                output_labels = [f'Stage Cost, $\overline{{l}}(z_k)$' for i in input_dims]

            title = ('Next State Error' if func['model_error'] else 'Next State') if is_state_gp \
                else ('Stage Cost Error' if func['model_error'] else 'Stage Cost')
            points_filename = (f'next_state_error_({device_idx}, {state_idx})_points' if func['model_error']
                               else f'next_state_({device_idx}, {state_idx})_points') if is_state_gp \
                else (f'stage_cost_error_{device_idx}_points' if func['model_error']
                      else f'stage_cost_{device_idx}_points')
            fig_filename = (f'next_state_error_({device_idx}, {state_idx})_gp'
                            if func['model_error'] else f'next_state_{d}_gp') if is_state_gp \
                else (f'stage_cost_error_{device_idx}_gp' if func['model_error'] else f'stage_cost_{device_idx}_gp')

            # plot gp approximation function values at the test inputs
            # plot true function values at the test inputs

            # fetch the predicted error is using the next state error gp,
            # else the predicted next state if using the next state gp
            y_pred, y_std = self.predict(x_test)

            if not run_mpc:

                if not os.path.exists(f'./{simulation_dir}/results/gp_results.csv'):
                    gp_results_df = pd.DataFrame(
                        columns=['Prediction Name', 'Function', 'No. Training Samples', 'Length Scale',
                                 'Output Variance', 'Measurement Noise', 'Score'])
                else:
                    gp_results_df = pd.read_csv(f'./{simulation_dir}/results/gp_results.csv', engine='python',
                                                index_col=0)
                    gp_results_df.drop(gp_results_df.loc[
                                           gp_results_df['Prediction Name'] == f'GP Ntr-{n_training_samples}'].index,
                                       inplace=True)

                figs, axes = self.plot(x_test, y_true, y_pred, y_std,
                                     input_dims, input_labels, output_labels, title,
                                     independent_data=independent_data)

                for ff, fig in enumerate(figs):
                    fig.savefig(f'{results_dir}/{fig_filename}_{ff}')

                self.score(y_true, y_pred)
                points_df = self.generate_points_df(x_test)
                points_df.to_csv(f'{results_dir}/{points_filename}')

                gp_results = {'Prediction Name': f'GP Ntr-{n_training_samples}',
                              'Function': f'Next State ({device_idx}, {state_idx})',
                              'No. Training Samples': n_training_samples,
                              'Length Scale': func['length_scale'],
                              'Output Variance': func['output_variance'],
                              'Measurement Noise': func['meas_noise'],
                              'Score': self.scores[0]}

                existing_row_indices = (gp_results_df['Prediction Name'] == simulation_name) \
                                       & (gp_results_df['Function'] == gp_results['Function'])

                if existing_row_indices.any(axis=0):
                    gp_results_df.loc[existing_row_indices] = gp_results
                else:
                    gp_results_df = gp_results_df.append(gp_results, ignore_index=True)
                    gp_results_df = gp_results_df.reindex()

                gp_results_df.to_csv(f'./{simulation_dir}/results/gp_results.csv')

                score_fig, score_ax = self.plot_score(gp_results_df.loc[gp_results_df['Function'] == gp_results['Function']])

                score_fig.savefig(f'{results_dir}/{fig_filename}_score')


class NextStateGP(GP):

    def __init__(self, state_output_variances, state_meas_noises, state_length_scales,
                 n_training_samples, n_test_samples,
                 state_lag, input_lag, disturbance_lag, output_lag,
                 n_states, n_inputs, n_disturbances, n_outputs):
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_disturbances = n_disturbances
        self.n_outputs = n_outputs

        self.state_lag = state_lag
        self.input_lag = input_lag
        self.disturbance_lag = disturbance_lag
        self.output_lag = output_lag

        super().__init__(n_training_samples=n_training_samples,
                         n_test_samples=n_test_samples,
                         n_input_dims=
                         n_states * (self.state_lag + 1)
                         + n_inputs * (self.input_lag + 1)
                         + n_disturbances * (self.disturbance_lag + 1),
                         n_output_dims=1,
                         output_dim=0)

        self.output_variances = [state_output_variances]
        self.meas_noises = [state_meas_noises]
        self.length_scales = np.array([state_length_scales])

        self.kernels = [ov * RBF(length_scale=ls)
                        for ov, ls in zip(self.output_variances, self.length_scales)]  # Matern(length_scale=2, nu=1.5)


class StageCostGP(GP):

    def __init__(self, cost_output_variance, cost_meas_noise, cost_length_scale,
                 n_training_samples, n_test_samples,
                 n_states, n_inputs, n_disturbances):
        super().__init__(n_training_samples=n_training_samples,
                         n_test_samples=n_test_samples,
                         n_input_dims=n_states + n_inputs + n_disturbances,
                         n_output_dims=1,
                         output_dim=0)

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.output_variances = [cost_output_variance]
        self.meas_noises = [cost_meas_noise]
        self.length_scales = np.array([cost_length_scale])

        self.kernels = [self.output_variances[d] * RBF(length_scale=self.length_scales[d]) for d in range(1)]