import numpy as np
from scipy.optimize import minimize, fsolve
import itertools

class EllipsoidApproximation:
    def __init__(self, model, beta, sigma_func, L_grad_h, L_g):
        self.beta = beta
        self.sigma_func = sigma_func
        self.L_grad_h = L_grad_h
        self.L_g = L_g
        self.model = model

    def get_z_bar(self, R, z):
        u = z[self.model.n_states:self.model.n_states + self.model.n_inputs]
        w = z[-self.model.n_disturbances:]
        z_bar = np.concatenate([R.p, u, w])
        return z_bar

    def linearized_next_state_func(self, z, z_bar):
        # \tilde{f}_\mu
        h_lin = self.model.linarized_next_state_prior_func(z, z_bar)
        f_lin = self.model.state_var_func(z_bar, return_std=False)
        return h_lin + f_lin

    def calc_linearized_next_state_ellipsoid(self, R, z):
        # \tilde{f}_mu
        z_bar = self.get_z_bar(R, z)
        p = self.model.next_state_prior_func(z_bar) + self.model.state_var_func(z_bar, return_std=False)
        A = self.model.linearized_next_state_prior_jacobian(z_bar)
        Q = A @ R.Q @ A.T
        return Ellipsoid(p, Q)

    def calc_hyperrect(self, R, z):
        # upperbounds of approximation error between true next state function and linearized next state function
        z_bar = self.get_z_bar(R, z)
        max_norm = R.calc_max_norm()
        return self.beta * self.sigma_func(z_bar) + (self.L_grad_h / 2) * max_norm**2 + (self.L_g / 2) * max_norm

    def calc_hyperrect_ellipsoid(self, c, d):
        # E(c, Q_\tilde{d})
        p = c
        Q = np.sqrt(self.model.n_states) * np.diag(d)
        return Ellipsoid(p, Q)

    def next_state_ellipsoid(self, R, z):
        # \tilde{m}
        d_tilde = self.calc_hyperrect(R, z)
        linearized_next_state_ellipsoid = self.calc_linearized_next_state_ellipsoid(R, z)
        hyperrect_ellipsoid = self.calc_hyperrect_ellipsoid(np.zeros_like(d_tilde), d_tilde)
        c = np.sqrt(np.trace(linearized_next_state_ellipsoid.Q) / np.trace(hyperrect_ellipsoid.Q))
        return linearized_next_state_ellipsoid.minkowski_sum(hyperrect_ellipsoid, c)



class Ellipsoid:
    def __init__(self, p, Q):
        self.p = p
        self.Q = Q
        self.Q_inv = np.linalg.inv(self.Q)
        self.n_states = self.p.shape[0]

    def affine_transformation(self, A, b):
        p = A @ self.p + b
        Q = A @ self.Q @ A.T
        return Ellipsoid(p, Q)

    def minkowski_sum(self, e_2, c):
        p = self.p + e_2.p
        Q = ((1 + (1 / c)) @ self.Q) + ((1 + c) @ e_2.Q)
        return Ellipsoid(p, Q)

    def calc_max_norm(self):
        # func = lambda z: -np.linalg.norm(z - z_bar, 2)
        # con = lambda z: (z[:n_states] - R.p).T @ np.linalg.inv(R.Q) @ (z[:n_states] - R.p) <= 1
        # res_opt = minimize(func, np.zeros_like(z_bar), constraints=[{'fun': con, 'type': 'ineq'}])

        S = np.eye(self.n_states)
        func = lambda s: -s.T @ S.T @ S @ s
        con = lambda s: -(s.T @ self.Q_inv @ s - 1)
        res_opt = minimize(func, np.zeros(self.n_states), constraints=[{'fun': con, 'type': 'ineq'}])
        return res_opt.x

    def is_subset(self, set_bounds):
        # perimeter_func = lambda x: (x - self.p).T @ self.Q_inv @ (x - self.p) - 1
        extrema = [self.p + sign * np.sqrt(np.diag(self.Q)) for sign in [-1, 1]]
        if np.all(extrema[0] >= set_bounds[:, 0]) and np.all(extrema[1] <= set_bounds[:, 1]):
            return True
        else:
            return False
        # x = fsolve(perimeter_func, self.p)

    def ineq_constraint(self, func):
        return self.eval_func(func)

    def eval_func(self, func):
        n_dims = self.p.shape[0]
        comb = np.array(list(itertools.combinations(np.concatenate([[-1, 1] for i in range(n_dims)]), n_dims)))
        extrema = [self.p + np.array(sign) * np.sqrt(np.diag(self.Q)) for sign in comb]
        return [func(x) for x in extrema]