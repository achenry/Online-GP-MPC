import numpy as np
from numpy.linalg import norm
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, fsolve
import matplotlib.pyplot as plt
from helper_functions import gradient_func, jacobian_func


class PDOptimizationMethod:

    def __init__(self, ocp):

        self.cost_trajectory = None
        self.dual_ineq_vars_trajectory = None
        self.dual_eq_vars_trajectory = None
        self.primal_vars_trajectory = None
        self.dual_ineq_vars_change_trajectory = None
        self.dual_eq_vars_change_trajectory = None
        self.primal_vars_change_trajectory = None

        self.regret = None
        self.ave_regret = None

        self.ocp = ocp
        self.n_primal_vars = self.ocp.n_optimization_vars
        # implicit convex feasible sets
        self.n_imp_dual_eq_vars = len(self.ocp.imp_eq_constraints)
        self.n_imp_dual_ineq_vars = len(self.ocp.imp_ineq_constraints)
        self.dual_ineq_constraint = []
        self.dual_eq_constraint = []
        self.n_dual_ineq_vars = self.ocp.n_exp_ineq_constraints
        self.n_dual_eq_vars = self.ocp.n_exp_eq_constraints

        self.set_dual_constraints()

        self.input_indices = [range((k * self.ocp.n_xustage_vars), (k * self.ocp.n_xustage_vars) + self.ocp.n_inputs)
                              for k in range(self.ocp.n_horizon)]

        self.state_indices = [range((k * self.ocp.n_xustage_vars) + self.ocp.n_inputs,
                                    (k * self.ocp.n_xustage_vars) + self.ocp.n_inputs + self.ocp.n_states)
                              for k in range(self.ocp.n_horizon)]

    def primal_ineq_constraint_func(self, primal_vars):
        # must negate function, because default
        return np.concatenate([-func['fun'](primal_vars) for func in self.ocp.exp_ineq_constraints]) \
            if self.ocp.n_exp_ineq_constraints else np.array([])

    def primal_eq_constraint_func(self, primal_vars):
        return np.concatenate([func['fun'](primal_vars) for func in self.ocp.exp_eq_constraints])

    def dual_ineq_constraint_func(self, dual_vars):
        return -dual_vars

    def set_dual_constraints(self):
        self.dual_ineq_constraint.append({'type': 'ineq', 'fun': self.dual_ineq_constraint_func})

    def lagrangian_gradient(self, y, x, eq_cons, ineq_cons):

        primal_vars = lambda y: y[:self.n_primal_vars]
        horizon_cost = lambda y: norm(primal_vars(y) - np.array(x), ord=2)
        imp_dual_ineq_vars = lambda y: y[self.n_primal_vars + self.n_imp_dual_eq_vars:]
        imp_ineq_funcs = lambda y: np.concatenate([con['fun'](primal_vars(y)) for con in ineq_cons])
        imp_dual_eq_vars = lambda y: y[self.n_primal_vars:self.n_primal_vars + self.n_imp_dual_eq_vars]
        imp_eq_funcs = lambda y: np.concatenate([con['fun'](primal_vars(y))
                                                 for con in eq_cons])

        lagrangian = lambda y: \
            horizon_cost(y) \
            + np.dot(imp_dual_eq_vars(y), imp_eq_funcs(y)) + \
            + np.dot(imp_dual_ineq_vars(y), imp_ineq_funcs(y))
        # self.primal_ineq_constraint_func(y[:self.n_primal_vars])) + \
        # np.dot(y[self.n_primal_vars + self.n_dual_ineq_vars:],
        # self.primal_eq_constraint_func(y[:self.n_primal_vars]))
        return gradient_func(lagrangian, y)

    def proj(self, x, eq_set_cons, ineq_set_cons, is_primal=False, is_dual=False):
        # TODO check for different solvers: SLSQP, trust-constr and smaller tolerance for SLSQP
        res_opt = x
        if self.ocp.imp_bounds_only and is_primal:
            z_tau = x[:self.n_primal_vars]
            for k in range(self.ocp.n_horizon):
                # project inputs analytically

                z_stage = np.hstack([self.ocp.x_stage(z_tau, k), self.ocp.u_stage(z_tau, k), self.ocp.w_stage(k)])

                res_opt[self.input_indices[k]] = np.min([np.max([res_opt[self.input_indices[k]],
                                                                 self.ocp.stage_bounds(z_stage)[self.ocp.n_states:, 0]
                                                                 * np.ones(self.ocp.n_inputs)], axis=0),
                                                         self.ocp.stage_bounds(z_stage)[self.ocp.n_states:, 1]
                                                         * np.ones(self.ocp.n_inputs)], axis=0)
                if k > 0:
                    res_opt[self.state_indices[k - 1]] = np.min([np.max([res_opt[self.state_indices[k - 1]],
                                                                         self.ocp.stage_bounds(z_stage)[
                                                                         :self.ocp.n_states, 0]
                                                                         * np.ones(self.ocp.n_states)], axis=0),
                                                                 self.ocp.stage_bounds(z_stage)[:self.ocp.n_states, 1]
                                                                 * np.ones(self.ocp.n_states)], axis=0)

            x_term = self.ocp.next_states(z_tau, -1)
            res_opt[self.state_indices[-1]] = np.min([np.max([res_opt[self.state_indices[-1]],
                                                              self.ocp.term_bounds(x_term)[:, 0]
                                                              * np.ones(self.ocp.n_states)], axis=0),
                                                      self.ocp.term_bounds(x_term)[:, 1]
                                                      * np.ones(self.ocp.n_states)], axis=0)

        elif is_dual:
            res_opt = np.max([res_opt, np.zeros(self.n_dual_ineq_vars)], axis=0)
        else:
            # res_opt = minimize(lambda y: norm(y - np.array(x).T, ord=2), x0=x, constraints=eq_set_cons + ineq_set_cons,
            #                    options={'maxiter': 1000}, method=None, tol=1e-10).x
            res_opt = fsolve(self.lagrangian_gradient,
                             x0=np.concatenate([x, np.zeros(self.n_imp_dual_eq_vars + self.n_imp_dual_ineq_vars)]),
                             xtol=1e-10, args=(x, eq_set_cons, ineq_set_cons))[:self.n_primal_vars]
        return res_opt

    def optimize(self, func, q_init, **options):

        """
        solve at each time-step tau
        disp : bool
        Set to True to print convergence messages.

        maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.

        return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

        initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.

        xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.

        fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
        """

        lambda_prior = options['lambda_prior']
        mu_prior = options['mu_prior']
        alpha = options['alpha']
        eta_ineq = options['eta_ineq']
        eta_eq = options['eta_eq']
        eps = options['eps']
        maxiter = options['maxiter']
        xtol = options['xtol']
        maxiter_plus = maxiter #1000
        k0 = options['args'][0]

        primal_vars = q_init[:self.n_primal_vars]
        dual_ineq_vars = q_init[self.n_primal_vars:self.n_primal_vars + self.n_dual_ineq_vars]
        dual_eq_vars = q_init[self.n_primal_vars + self.n_dual_ineq_vars:]
        q_opt = np.hstack([primal_vars, dual_ineq_vars, dual_eq_vars])

        primal_vars_new = primal_vars
        dual_ineq_vars_new = dual_ineq_vars
        dual_eq_vars_new = dual_eq_vars
        q_opt_new = q_opt

        primal_vars_trajectory = [np.array(primal_vars_new)]
        dual_eq_vars_trajectory = [np.array(dual_eq_vars_new)]
        dual_ineq_vars_trajectory = [np.array(dual_ineq_vars_new)]
        cost_trajectory = [self.ocp.horizon_cost(primal_vars_new, k0=k0)]

        primal_vars_change_trajectory = []
        dual_eq_vars_change_trajectory = []
        dual_ineq_vars_change_trajectory = []

        online_tau = 0
        tau = 0
        is_online = True
        while 1:
        # while tau < maxiter:

            alpha_vec = alpha  # np.ones(self.n_primal_vars) * alpha
            # alpha_vec[0] = 0.005

            # np.sum((self.ocp.horizon_cost_jacobian(primal_vars, k0=k0) - jacobian_func(self.ocp.horizon_cost, primal_vars))**2)
            # np.sum((self.ocp.next_state_constraint_jacobian(primal_vars, k0) - jacobian_func(self.ocp.next_state_constraint_func, primal_vars))**2)
            # np.sum((self.ocp.stage_ineq_constraint_jacobian(primal_vars, k0) - jacobian_func(self.ocp.stage_ineq_constraint_func, primal_vars))**2)

            primal_vars_new = self.proj(primal_vars -
                                        alpha_vec * (self.ocp.horizon_cost_jacobian(primal_vars, k0=k0)[0]
                                                     + np.dot(
                        self.ocp.stage_ineq_constraint_jacobian(primal_vars, k0=k0).T, dual_ineq_vars)
                                                     + np.dot(
                        self.ocp.next_state_constraint_jacobian(primal_vars, k0=k0).T,
                        dual_eq_vars)), self.ocp.imp_eq_constraints, self.ocp.imp_ineq_constraints, is_primal=True)

            cost_new = self.ocp.horizon_cost(primal_vars_new, k0=k0)

            if dual_ineq_vars.shape[0]:
                dual_ineq_vars_new = self.proj(dual_ineq_vars +
                                               (eta_ineq * alpha *
                                                (self.ocp.stage_ineq_constraint_func(primal_vars_new, k0=k0)
                                                 - (eps * (dual_ineq_vars - lambda_prior)))),
                                               [], self.dual_ineq_constraint, is_dual=True)

            if dual_eq_vars.shape[0]:
                # eta_vec = eta # np.ones(self.n_dual_eq_vars) * eta
                # eta_vec[1] = 0.1
                dual_eq_vars_new = dual_eq_vars + (eta_eq * alpha *
                                                   (self.ocp.next_state_constraint_func(primal_vars_new, k0=k0)
                                                    - (eps * (dual_eq_vars - mu_prior))))

            if is_online:
                primal_vars_trajectory.append(primal_vars_new)
                dual_eq_vars_trajectory.append(dual_eq_vars_new)
                dual_ineq_vars_trajectory.append(dual_ineq_vars)
                cost_trajectory.append(cost_new)
                primal_vars_change_trajectory.append(norm(primal_vars_new - primal_vars))
                dual_eq_vars_change_trajectory.append(norm(dual_eq_vars_new - dual_eq_vars))
                dual_ineq_vars_change_trajectory.append(norm(dual_ineq_vars_new - dual_ineq_vars))

                if tau == maxiter - 1:
                    is_online = False
                    online_tau = tau

            # if cost_trajectory[-1] > 50:
            #     print(1)
            # TODO check GP derivative of F, L and see if it needs to be clipped
            # TODO check dual_eq_vars and see if it needs to be projected onto closed set
            # TODO check variance of state gp predictions. If large can't trust gp.

            # to convergence
            q_opt_new = np.hstack([primal_vars_new, dual_ineq_vars_new, dual_eq_vars_new])
            if (norm((q_opt_new - q_opt), 2) < xtol) or (tau == maxiter_plus):
                break

            primal_vars = primal_vars_new
            dual_ineq_vars = dual_ineq_vars_new
            dual_eq_vars = dual_eq_vars_new
            q_opt = q_opt_new

            tau += 1

        self.regret = cost_trajectory[-1] - cost_new
        self.ave_regret = self.regret / (k0 + 1)

        self.primal_vars_change_trajectory = primal_vars_change_trajectory
        self.dual_ineq_vars_change_trajectory = dual_ineq_vars_change_trajectory
        self.dual_eq_vars_change_trajectory = dual_eq_vars_change_trajectory
        self.primal_vars_trajectory = primal_vars_trajectory
        self.dual_ineq_vars_trajectory = dual_ineq_vars_trajectory
        self.dual_eq_vars_trajectory = dual_eq_vars_trajectory
        self.cost_trajectory = cost_trajectory

        cost = cost_trajectory[-1]  # np.append(cost, self.ocp.horizon_cost(primal_vars))

        res_opt = OptimizeResult(x=q_opt_new, fun=cost, nit=(online_tau + 1))
        return res_opt

    def plot(self):
        n_plots = 3 if self.ocp.n_exp_ineq_constraints else 2
        conv_fig, conv_ax = plt.subplots(n_plots, 1, sharex=True, frameon=False)
        conv_fig.align_ylabels()

        iters = np.arange(len(self.primal_vars_trajectory)).astype('int')
        iter_interval = len(iters) / 5
        conv_ax[-1].set_xlabel('$\\tau$')
        conv_ax[0].plot(iters[1:], self.primal_vars_change_trajectory)
        conv_ax[0].set_ylabel('$\left \Vert \mathbf{{z_{\\tau  + 1}}} - \mathbf{{z_{\\tau}}} \\right \Vert_2$',
                              rotation=0)
        conv_ax[1].plot(iters[1:], self.dual_eq_vars_change_trajectory)
        conv_ax[1].set_ylabel('$\left \Vert \mathbf{{\mu_{\\tau  + 1}}} - \mathbf{{\mu_{\\tau}}} \\right \Vert_2$',
                              rotation=0)

        if self.ocp.n_exp_ineq_constraints:
            conv_ax[2].plot(iters[1:], self.dual_ineq_vars_change_trajectory)
            conv_ax[2].set_ylabel(
                '$\left \Vert \mathbf{{\lambda_{\\tau  + 1}}} - \mathbf{{\lambda_{\\tau}}} \\right \Vert_2$',
                rotation=0)

        for ax in conv_ax:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(bbox_to_anchor=(1, 1))
            # ax.set_xticks(np.arange(0, len(iters), iter_interval))

        conv_fig.show()
        self.conv_fig = conv_fig
        self.conv_ax = conv_ax

        n_plots = 4 if self.ocp.n_exp_ineq_constraints else 3
        var_fig, var_ax = plt.subplots(n_plots, 1, sharex=True, frameon=False)
        var_fig.align_ylabels()
        var_ax[-1].set_xlabel('$\\tau$')
        primal_vars_trajectory = np.vstack(self.primal_vars_trajectory)
        for v in range(self.n_primal_vars):
            var_ax[0].plot(iters, primal_vars_trajectory[:, v], label=f"$\mathbf{{z_{{{v}}}}}$")
        var_ax[0].set_ylabel('$\mathbf{z_{\\tau}}$', rotation=0)

        dual_eq_vars_trajectory = np.vstack(self.dual_eq_vars_trajectory)
        for v in range(self.n_dual_eq_vars):
            var_ax[1].plot(iters, dual_eq_vars_trajectory[:, v], label=f"$\mathbf{{\mu_{{{v}}}}}$")
        var_ax[1].set_ylabel('$\mathbf{\mu_{\\tau}}$', rotation=0)

        ax_idx = 2
        if self.ocp.n_exp_ineq_constraints:
            dual_ineq_vars_trajectory = np.vstack(self.dual_ineq_vars_trajectory)
            for v in range(self.n_dual_ineq_vars):
                var_ax[ax_idx].plot(iters, dual_ineq_vars_trajectory[:, v], label=f"$\mathbf{{\lambda_{{{v}}}}}$")
            var_ax[ax_idx].set_ylabel('$\mathbf{\lambda_{\\tau}}$', rotation=0)
            ax_idx += 1

        var_ax[ax_idx].plot(iters, self.cost_trajectory)
        var_ax[ax_idx].set_ylabel('$L_{\\tau}$', rotation=0)

        for ax in var_ax:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if False and len(ax.get_lines()) <= 15:
                ax.legend(bbox_to_anchor=(1, 1), ncol=3)
            # ax.set_xticks(np.arange(0, len(iters), iter_interval))

        var_fig.show()
        self.var_fig = var_fig
        self.var_ax = var_ax

        return self.conv_fig, self.conv_ax, self.var_fig, self.var_ax
