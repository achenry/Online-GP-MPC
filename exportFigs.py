import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from mpl_settings import *


def normal_dist(x, mean, sd):
    prob_density = (1 / (2 * np.pi * sd ** 2)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


def multi_normal_dist(X, mean, cov):
    prob_density = (2 * np.pi) ** (-X.shape[1] / 2) \
                   * np.linalg.det(cov) ** (-0.5) \
                   * np.exp(-0.5 * (X - mean) @ np.linalg.inv(cov) @ (X - mean).T)
    return prob_density


def exportGPs():
    path = './gd_plots'
    if not os.path.exists(path):
        os.mkdir(path)

    X_test = np.linspace(0, 10, 100)
    y_pred = np.zeros(100)
    y_std = np.ones(100)
    f = np.sin(X_test)
    fig_0, ax_0 = plt.subplots(1, 1)
    ax_0.plot(X_test, f, color='tab:blue', label='True Function')
    ax_0.plot(X_test, y_pred, color='tab:red', label='GP Posterior Mean')
    ax_0.fill_between(X_test, (y_pred - 2 * y_std), (y_pred + 2 * y_std), color='coral',
                      label='$\pm2*\sigma$')
    ax_0.legend(bbox_to_anchor=(1, 1))

    ax_0.spines['top'].set_visible(False)
    ax_0.spines['right'].set_visible(False)
    ax_0.set_ylim([-2, 2])
    fig_0.show()
    fig_0.savefig(f'{path}/prior_gp.png')

    X = np.random.uniform(0, 10, (3, 1))
    y = np.sin(X) + np.random.normal(0, 0.01, (3, 1))
    gp_fit = GaussianProcessRegressor(kernel=1 * RBF(1)).fit(X, y)
    X_test = np.linspace(0, 10, 100).reshape((100, 1))
    f = np.sin(X_test)
    y_pred, y_std = gp_fit.predict(X_test, return_std=True)
    fig_1, ax_1 = plt.subplots(1, 1)
    ax_1.plot(X_test[:, 0], f[:, 0], color='tab:blue', label='True Function')
    ax_1.plot(X_test[:, 0], y_pred[:, 0], color='tab:red', label='GP Posterior Mean')
    ax_1.fill_between(X_test[:, 0], (y_pred[:, 0] - 2 * y_std), (y_pred[:, 0] + 2 * y_std), color='coral',
                      label='$\pm2*\sigma$')
    ax_1.scatter(X, y, color='tab:blue', label='Training Data')
    ax_1.legend(bbox_to_anchor=(1, 1))

    ax_1.spines['top'].set_visible(False)
    ax_1.spines['right'].set_visible(False)
    ax_1.set_ylim([-2, 2])
    fig_1.show()
    fig_1.savefig(f'{path}/posterior_gp_0.png')

    X = np.random.uniform(0, 10, (12, 1))
    y = np.sin(X) + np.random.normal(0, 0.01, (12, 1))
    gp_fit = GaussianProcessRegressor(kernel=1 * RBF(1)).fit(X, y)
    X_test = np.linspace(0, 10, 100).reshape((100, 1))
    f = np.sin(X_test)
    y_pred, y_std = gp_fit.predict(X_test, return_std=True)
    fig_2, ax_2 = plt.subplots(1, 1)
    ax_2.plot(X_test[:, 0], f[:, 0], color='tab:blue', label='True Function')
    ax_2.plot(X_test[:, 0], y_pred[:, 0], color='tab:red', label='GP Posterior Mean')
    ax_2.fill_between(X_test[:, 0], (y_pred[:, 0] - 2 * y_std), (y_pred[:, 0] + 2 * y_std), color='coral',
                      label='$\pm2*\sigma$')
    # plot training data
    ax_2.scatter(X, y, color='tab:blue', label='Training Data')

    ax_2.legend(bbox_to_anchor=(1, 1))

    ax_2.spines['top'].set_visible(False)
    ax_2.spines['right'].set_visible(False)
    ax_2.set_ylim([-2, 2])
    fig_2.show()
    fig_2.savefig(f'{path}/posterior_gp_1.png')


def exportGDs():
    path = './gd_plots'
    if not os.path.exists(path):
        os.mkdir(path)

    n_samples = 1000
    mean = 0
    cov = 1
    x = np.sort(np.random.normal(mean, cov, n_samples))
    uni_dist = normal_dist(x, mean, cov)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, uni_dist)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(f'{path}/uni_gd.png')

    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    n_samples = 1000
    X = np.random.multivariate_normal(mean, cov, n_samples)
    multi_dist = multi_normal_dist(X, mean, cov)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.xaxis.set_tick_params(pad=20)
    ax.yaxis.set_tick_params(pad=20)
    ax.zaxis.set_tick_params(pad=20)
    ax.plot_trisurf(X[:, 0], X[:, 1], multi_dist[np.diag_indices(n_samples)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0)
    fig.show()
    fig.savefig(f'{path}/multi_gd.png', pad_inches=0)


def exportSublinear():
    path = './gd_plots'
    if not os.path.exists(path):
        os.mkdir(path)

    x = np.linspace(1, 1000, 1000)
    y = np.log(x) + 2
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel('$\\tau$')
    ax.set_ylabel('$R_{MaxIter}(\mathcal{A})$', rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(f'{path}/sublinear.png')


def main():
    exportGDs()
    # exportGPs()
    # exportSublinear()


if __name__ == '__main__':
    main()
