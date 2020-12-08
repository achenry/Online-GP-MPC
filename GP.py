import numpy as np
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pylab as plt
from mpl_settings import *
from time import sleep
import pandas as pd


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
            gradients.append(-(1 / self.length_scales[d] ** 2)
                             * (x_test - self.x_train).T
                             @ np.array([cov_train_test[s] * alpha[s] for s in range(self.n_training_samples)]))

        return np.array(gradients).reshape((self.n_output_dims, self.n_input_dims))

    def covariance_jacobian(self, x_test):
        # TODO test
        # calculates derivative of covariance = kernel
        gradients = []
        for d in range(self.n_output_dims):
            cov_train_test = self.kernels[d](self.x_train, [x_test])
            gradients.append((1 / self.length_scales[d] ** 2)
                             * np.array([cov_train_test[s, :] * (self.x_train[s, :] - x_test)
                                         for s in range(self.n_training_samples)]))

    def update_inv_cov_train(self):

        for d in range(self.n_output_dims):
            cov_train = self.kernels[d](self.x_train)
            if self.meas_noises[d]:
                cov_train[np.diag_indices_from(cov_train)] += self.meas_noises[d]

            self.inv_cov_train[d] = np.linalg.inv(cov_train)

    def sq_exp_cov(self, x, y, output_variance, length_scale):
        return output_variance * np.exp(-(np.subtract.outer(x, y) ** 2) / (2 * length_scale ** 2))

    def score(self, y_true, y_pred):

        #for d in range(self.n_output_dims):
        res_sum_of_squares = ((y_true - y_pred)**2).sum(axis=0, dtype=np.float64)
        total_sum_of_squares = ((y_true - y_true.mean(axis=0))**2).sum(axis=0, dtype=np.float64)

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
            # if self.x_train.shape[0]:

            cov_test_train = self.kernels[d](x_test, self.x_train)
            cov_test = self.kernels[d](x_test, x_test)

            cov_inv = self.inv_cov_train[d]

            pred.append((cov_test_train @ cov_inv @ self.y_train[:, d])[:, np.newaxis])
            cov = cov_test - (cov_test_train @ cov_inv @ cov_test_train.T)

            std.append(np.sqrt(np.diag(cov))[:, np.newaxis])

        pred = np.hstack(pred)
        std = np.hstack(std)

        # self.x_test = np.vstack([self.x_test, x_test])
        # self.y_pred = np.vstack([self.y_pred, pred])
        # self.y_std = np.vstack([self.y_std, std])

        # new_df = pd.DataFrame({'x_test': list(x_test),
        #                        'y_pred': list(pred),
        #                        'y_std': list(std)})
        # df = df.append(new_df)

        return pred, std

    def set_training_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def collect_training_data(self, df, n_samples=1):

        # fetch new samples from db
        n_samples = min([n_samples, len(df.index)]) if type(n_samples) is int else len(df.index)

        new_samples = df.tail(n_samples)  # get(doc_id=len(next_state_table))

        # drop fetched samples from training data db
        df.drop(new_samples.index, inplace=True)  # remove(doc_ids=[len(next_state_table)])

        # add new samples to training dataset
        x_train = np.vstack(new_samples['x_train'].values)
        y_train = np.vstack(new_samples['y_train'].values) #[:, self.output_dim][:, np.newaxis]

        # collect moving window of training samples
        self.x_train = np.vstack([self.x_train, x_train])
        n_training_samples = self.x_train.shape[0]
        self.x_train = self.x_train[-min(self.n_training_samples, n_training_samples):, :]
        self.y_train = np.vstack([self.y_train, y_train])[-min(self.n_training_samples, n_training_samples):, :]

        self.update_inv_cov_train()


    def collect_training_data_thread_func(self, df, is_simulation_running, sampling_period):
        while not is_simulation_running.is_set():
            self.collect_training_data(df, 1)
            sleep(sampling_period)

    def plot_score(self, gp_df):
        fig, ax = plt.subplots(1, 1, frameon=False)
        fig.suptitle(f'GP Approximation Score $= 1 - \\frac{{RSS}}{{TSS}}$')
        ax.plot(gp_df['No. Training Samples'].values, gp_df['Score'].values)
        fig.show()
        return fig, ax

    def plot(self, x_test, y_actual, y_pred, y_std,
             input_dims, input_labels, output_dims, output_labels, title):
        figs = []
        axes = []
        for d in range(len(input_labels)):
            # plot based on first dim of y la only
            fig, ax = plt.subplots(1, 1, frameon=False)
            figs.append(fig)
            axes.append(ax)
            fig.suptitle(title)

            #if self.n_output_dims == 1:
             #   ax = np.array([ax])

            ax.set_xlabel(input_labels[self.output_dim])
            ax.set_ylabel(output_labels[self.output_dim], rotation=0)
            ax.set_xlabel(input_labels[self.output_dim])

            ax.scatter(self.x_train[:, input_dims[d]], self.y_train[:, self.output_dim],
                       label='Training Data')

            ax.plot(x_test[:, input_dims[d]], y_pred[:, self.output_dim], 'red',
                    label='GP Approximation')

            ax.fill_between(x_test[:, input_dims[d]],
                            (y_pred - y_std)[:, self.output_dim],
                            (y_pred + y_std)[:, self.output_dim], alpha=0.2)

            ax.plot(x_test[:, input_dims[d]], y_actual, #[:, output_dims[self.output_dim]],
                    label='Actual Function')

            ax.legend(bbox_to_anchor=(1, 1))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.show()

        return figs, axes

    def generate_results_df(self, y_true, x_test):
        y_pred, y_std = self.predict(x_test)
        self.score(y_true, y_pred)

        dict = {
            'x_train': list(self.x_train),
            'y_train': list(self.y_train),
            'x_test': list(x_test),
            'y_pred': list(y_pred),
            'y_std': list(y_std),
            'score': list(self.scores)
        }
        df = pd.DataFrame({key: pd.Series(val) for key, val in dict.items()})

        return df


class NextStateGP(GP):

    def __init__(self, state_output_variances, state_meas_noises, state_length_scales,
                 n_training_samples, n_test_samples,
                 state_lag, input_lag, disturbance_lag,
                 n_states, n_inputs, n_disturbances, output_dim):
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_disturbances = n_disturbances

        self.state_lag = state_lag
        self.input_lag = input_lag
        self.disturbance_lag = disturbance_lag

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
        self.length_scales = [state_length_scales]

        self.kernels = [ov * RBF(length_scale=ls)
                        for ov, ls in zip(self.output_variances, self.length_scales)]  # Matern(length_scale=2, nu=1.5)


class CostGP(GP):

    def __init__(self, cost_output_variance, cost_meas_noise, cost_length_scale,
                 n_training_samples, n_test_samples,
                 n_states, n_inputs):

        super().__init__(n_training_samples=n_training_samples,
                         n_test_samples=n_test_samples,
                         n_input_dims=n_states + n_inputs,
                         n_output_dims=1,
                         output_dim=0)

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.output_variances = [cost_output_variance]
        self.meas_noises = [cost_meas_noise]
        self.length_scales = [cost_length_scale]

        self.kernels = [self.output_variances[d] * RBF(length_scale=self.length_scales[d]) for d in range(1)]
