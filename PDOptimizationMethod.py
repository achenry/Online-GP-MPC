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
        self.n_primal_vars = self.ocp.n_total_vars
        # implicit convex feasible sets
        self.n_imp_dual_eq_vars = len(self.ocp.imp_eq_constraints)
        self.n_imp_dual_ineq_vars = len(self.ocp.imp_ineq_constraints)
        self.dual_ineq_constraint = []
        self.dual_eq_constraint = []
        self.n_ineq_dual_vars = self.ocp.n_exp_ineq_constraints
        self.n_eq_dual_vars = self.ocp.n_exp_eq_constraints

        self.set_dual_constraints()

        self.input_indices = [range((k * self.ocp.n_stage_vars), (k * self.ocp.n_stage_vars) + self.ocp.n_inputs)
                              for k in range(self.ocp.n_horizon)]
        self.state_indices = [range((k * self.ocp.n_stage_vars) + self.ocp.n_inputs,
                                    (k * self.ocp.n_stage_vars) + self.ocp.n_inputs + self.ocp.n_states)
                              for k in range(self.ocp.n_horizon)]

    def primal_ineq_constraint_func(self, primal_vars):
        # must negate function, because default
        return np.concatenate([-func['fun'](primal_vars) for func in self.ocp.exp_ineq_constraints]) \
            if len(self.primal_ineq_constraint) else np.array([])

    def primal_eq_constraint_func(self, primal_vars):
        return np.concatenate([func['fun'](primal_vars) for func in self.ocp.exp_eq_constraints])

    def dual_ineq_constraint_func(self, dual_vars):
        return -dual_vars

    def set_dual_constraints(self):
        self.dual_ineq_constraint.append({'type': 'ineq', 'fun': self.dual_ineq_constraint_func})

    def lagrangian_gradient(self, y, x, eq_cons, ineq_cons):

        primal_vars = lambda y: y[:self.n_primal_vars]
        horizon_cost = lambda y: norm(primal_vars(y) - np.array(x), ord=2)
        imp_ineq_dual_vars = lambda y: y[self.n_primal_vars + self.n_imp_dual_eq_vars:]
        imp_ineq_funcs = lambda y: np.concatenate([con['fun'](primal_vars(y)) for con in ineq_cons])
        imp_eq_dual_vars = lambda y: y[self.n_primal_vars:self.n_primal_vars + self.n_imp_dual_eq_vars]
        imp_eq_funcs = lambda y: np.concatenate([con['fun'](primal_vars(y))
                                                 for con in eq_cons])

        lagrangian = lambda y: \
            horizon_cost(y) \
            + np.dot(imp_eq_dual_vars(y), imp_eq_funcs(y)) + \
            + np.dot(imp_ineq_dual_vars(y), imp_ineq_funcs(y))
        # self.primal_ineq_constraint_func(y[:self.n_primal_vars])) + \
        # np.dot(y[self.n_primal_vars + self.n_ineq_dual_vars:],
        # self.primal_eq_constraint_func(y[:self.n_primal_vars]))
        return gradient_func(lagrangian, y)

    def proj(self, x, eq_set_cons, ineq_set_cons):
        # TODO check for different solvers: SLSQP, trust-constr and smaller tolerance for SLSQP
        #

        res_opt = x
        if self.ocp.imp_bounds_only:
            for k in range(self.ocp.n_horizon):
                # project inputs analytically
                res_opt[self.input_indices[k]] = np.min([np.max([res_opt[self.input_indices[k]],
                                                                 self.ocp.stage_bounds[self.ocp.n_states:, 0]
                                                                 * np.ones(self.ocp.n_inputs)], axis=0),
                                                         self.ocp.stage_bounds[self.ocp.n_states:, 1]
                                                         * np.ones(self.ocp.n_inputs)], axis=0)
                if k > 0:
                    res_opt[self.state_indices[k - 1]] = np.min([np.max([res_opt[self.state_indices[k - 1]],
                                                                         self.ocp.stage_bounds[:self.ocp.n_states, 0]
                                                                         * np.ones(self.ocp.n_states)], axis=0),
                                                                 self.ocp.stage_bounds[:self.ocp.n_states, 1]
                                                                 * np.ones(self.ocp.n_states)], axis=0)
            res_opt[self.state_indices[-1]] = np.min([np.max([res_opt[self.state_indices[- 1]],
                                                              self.ocp.term_bounds[:, 0]
                                                              * np.ones(self.ocp.n_states)], axis=0),
                                                      self.ocp.term_bounds[:, 1]
                                                      * np.ones(self.ocp.n_states)], axis=0)

        else:
            # res_opt = minimize(lambda y: norm(y - np.array(x).T, ord=2), x0=x, constraints=eq_set_cons + ineq_set_cons,
            #                    options={'maxiter': 1000}, method=None, tol=1e-10).x
            res_opt = fsolve(self.lagrangian_gradient,
                             x0=np.concatenate([x, np.zeros(self.n_imp_dual_eq_vars + self.n_imp_dual_ineq_vars)]),
                             xtol=1e-10, args=(x, eq_set_cons, ineq_set_cons))[:self.n_primal_vars]
        return res_opt

    def optimize(self, func, z_init, **options):

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
        eta = options['eta']
        eps = options['eps']
        maxiter = options['maxiter']
        xtol = options['xtol']
        # number of time slots
        # T = options['T']

        primal_vars = z_init[:self.n_primal_vars]
        ineq_dual_vars = z_init[self.n_primal_vars:self.n_primal_vars + self.n_ineq_dual_vars]
        eq_dual_vars = z_init[self.n_primal_vars + self.n_ineq_dual_vars:]

        primal_vars_trajectory = []
        dual_eq_vars_trajectory = []
        dual_ineq_vars_trajectory = []
        cost_trajectory = []

        primal_vars_change_trajectory = []
        dual_eq_vars_change_trajectory = []
        dual_ineq_vars_change_trajectory = []

        for tau in range(maxiter):
            primal_vars_temp = np.array(primal_vars)
            ineq_dual_vars_temp = np.array(ineq_dual_vars)
            eq_dual_vars_temp = np.array(eq_dual_vars)

            primal_vars = self.proj(primal_vars_temp -
                                    alpha * (self.ocp.horizon_cost_jacobian(primal_vars_temp)[0]
                                             + np.dot(self.ocp.state_ineq_constraint_jacobian(primal_vars_temp).T,
                                                      ineq_dual_vars_temp)
                                             + np.dot(self.ocp.next_state_constraint_jacobian(primal_vars_temp).T,
                                                      eq_dual_vars_temp)),
                                    self.ocp.imp_eq_constraints, self.ocp.imp_ineq_constraints)

            if ineq_dual_vars.shape[0]:
                ineq_dual_vars = self.proj(ineq_dual_vars_temp +
                                           (eta * alpha *
                                            (self.primal_ineq_constraint_func(primal_vars_temp)
                                             - (eps * (ineq_dual_vars_temp - lambda_prior)))),
                                           [], self.dual_ineq_constraint)

            if eq_dual_vars.shape[0]:
                eq_dual_vars = eq_dual_vars_temp + (eta * alpha *
                                                    (self.primal_eq_constraint_func(primal_vars_temp)
                                                     - (eps * (eq_dual_vars_temp - mu_prior))))

            primal_vars_trajectory.append(np.array(primal_vars))
            dual_eq_vars_trajectory.append(np.array(eq_dual_vars))
            dual_ineq_vars_trajectory.append(np.array(ineq_dual_vars))
            cost_trajectory.append(self.ocp.horizon_cost(primal_vars))

            primal_vars_change_trajectory.append(norm(primal_vars - primal_vars_temp))
            dual_eq_vars_change_trajectory.append(norm(eq_dual_vars - eq_dual_vars_temp))
            dual_ineq_vars_change_trajectory.append(norm(ineq_dual_vars - ineq_dual_vars_temp))
            if norm((np.hstack([primal_vars, ineq_dual_vars, eq_dual_vars])
                     - np.hstack([primal_vars_temp, ineq_dual_vars_temp, eq_dual_vars_temp])), 2) < xtol:
                break

        self.regret = sum(cost_trajectory) - \
                 minimize(lambda z: np.sum(self.ocp.horizon_cost(z)), np.zeros(self.n_primal_vars),
                          jac=self.ocp.horizon_cost_jacobian, constraints=self.ocp.constraints).fun
        self.ave_regret = self.regret / tau

        self.primal_vars_change_trajectory = primal_vars_change_trajectory
        self.dual_ineq_vars_change_trajectory = dual_ineq_vars_change_trajectory
        self.dual_eq_vars_change_trajectory = dual_eq_vars_change_trajectory
        self.primal_vars_trajectory = primal_vars_trajectory
        self.dual_ineq_vars_trajectory = dual_ineq_vars_trajectory
        self.dual_eq_vars_trajectory = dual_eq_vars_trajectory
        self.cost_trajectory = cost_trajectory

        z_opt = np.hstack([primal_vars, ineq_dual_vars, eq_dual_vars])

        cost = self.ocp.horizon_cost(primal_vars)  # np.append(cost, self.ocp.horizon_cost(primal_vars))

        res_opt = OptimizeResult(x=z_opt, fun=cost, nit=tau)
        return res_opt

    def plot(self):
        conv_fig, conv_ax = plt.subplots(3, 1, sharex=True, frameon=False)
        conv_fig.align_ylabels()

        iters = range(len(self.primal_vars_trajectory))
        conv_ax[-1].set_xlabel('iteration count')
        conv_ax[0].plot(iters, self.primal_vars_change_trajectory)
        conv_ax[0].set_title('Primal Variable ($z$) L2-Norm Change Convergence')
        conv_ax[1].plot(iters, self.dual_eq_vars_change_trajectory)
        conv_ax[1].set_title('Equality Dual Variable ($\mu$) L2-Norm Change Convergence')
        conv_ax[2].plot(iters, self.dual_ineq_vars_change_trajectory)
        conv_ax[2].set_title('Inequality Dual Variable ($\lambda$) L2-Norm Change Convergence')

        for ax in conv_ax:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(bbox_to_anchor=(1, 1))

        conv_fig.show()
        self.conv_fig = conv_fig
        self.conv_ax = conv_ax

        var_fig, var_ax = plt.subplots(4, 1, sharex=True, frameon=False)
        var_fig.align_ylabels()
        var_ax[-1].set_xlabel('iteration count')
        primal_vars_trajectory = np.vstack(self.primal_vars_trajectory)
        for v in range(self.n_primal_vars):
            var_ax[0].plot(iters, primal_vars_trajectory[:, v], label=f"$z_{{{v}}}$")
        var_ax[0].set_title('Primal Variable ($z$) Convergence')

        dual_eq_vars_trajectory = np.vstack(self.dual_eq_vars_trajectory)
        for v in range(self.n_eq_dual_vars):
            var_ax[1].plot(iters, dual_eq_vars_trajectory[:, v], label=f"$\mu_{{{v}}}$")
        var_ax[1].set_title('Equality Dual Variable ($\mu$) Convergence')

        dual_ineq_vars_trajectory = np.vstack(self.dual_ineq_vars_trajectory)
        for v in range(self.n_ineq_dual_vars):
            var_ax[2].plot(iters, dual_ineq_vars_trajectory[:, v], label=f"$\lambda_{{{v}}}$")
        var_ax[2].set_title('Inequality Dual Variable ($\lambda$) Convergence')

        var_ax[3].plot(iters, self.cost_trajectory)
        var_ax[3].set_title('Cost Function ($l$) Convergence')

        for ax in var_ax:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(bbox_to_anchor=(1, 1), ncol=3)

        var_fig.show()
        self.var_fig = var_fig
        self.var_ax = var_ax

        return self.conv_fig, self.conv_ax, self.var_fig, self.var_ax
