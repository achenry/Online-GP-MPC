import numpy as np

class OCP:

    def __init__(self, n_horizon, state_lag, input_lag, disturbance_lag, output_lag, mpc_t_step,
                 imp_bounds_only=True, model_type='discrete'):

        # horizon length
        self.n_horizon = n_horizon
        self.true_disturbances = None

        # stage cost, terminal cost and dynamic state functions
        self.stage_cost_func = None
        self.terminal_cost_func = None
        self.next_state_func = None
        self.stage_ineq_constraint_f = None

        self.stage_cost_jacobian = None
        self.terminal_cost_jacobian = None
        self.next_state_jacobian = None
        self.stage_ineq_constraint_jac = None

        # all constraints
        self.constraints = []

        # constraints implicit in feasible sets
        self.imp_ineq_constraints = []
        self.imp_eq_constraints = []
        self.imp_bounds_only = imp_bounds_only
        self.stage_bounds = None
        self.term_bounds = None

        # constraints explicit in dual variables

        self.parameters = {}
        self.n_stage_vars = 0
        self.n_total_vars = 0
        self.n_optimization_vars = 0
        self.n_states = 0
        self.n_inputs = 0
        self.n_disturbances = 0
        self.n_outputs = 0
        self.n_xustage_vars = 0

        # the indices of the mpc primal variables which correspond to the current state at each time step
        self.x_stage_indices = None
        # the indices of the mpc primal variables which correspond to the current input at each time step
        self.u_stage_indices = None
        # the indices of the mpc primal variables which correspond to the current disturbance at each time step
        self.w_stage_indices = None
        self.y_stage_indices = None
        self.z_stage_indices = None
        # the indices of the mpc primal variables which correspond to the next state at each time step
        self.next_state_indices = None
        self.next_state_fullvar_indices = None
        # the indices of the mpc primal variables which correspond to the lagged states required at each time step
        self.x_lagged_indices = None
        # the indices of the mpc primal variables which correspond to the lagged inputs required at each time step
        self.u_lagged_indices = None
        # the indices of the mpc primal variables which correspond to the lagged disturbance required at each time step
        self.w_lagged_indices = None
        self.y_lagged_indices = None

        self.xustage_opt_indices = None
        self.xustage_full_indices = None
        self.xterm_opt_indices = None

        # indices corresponding to the stage cost input variables from optimization variables z
        self.stage_cost_fullvar_indices = None
        # indices corresponding to the optimization variables from the stage cost jacobian
        self.stage_cost_optvar_indices = None
        # indices corresponding to the state inequality input from optimization variables z
        self.stage_ineq_fullvar_indices = None
        # indices corresponding to the optimization variables from the state inequality jacobian
        self.stage_ineq_optvar_indices = None
        # indices corresponding to the next state equality input from optimization variables z
        self.next_state_eq_fullvar_indices = None
        # indices corresponding to the optimization variables from the next state equality jacobian
        self.next_state_eq_optvar_indices = None

        self.state_lag = state_lag
        self.input_lag = input_lag
        self.disturbance_lag = disturbance_lag
        self.output_lag = output_lag

        self.previous_x_lagged = None
        self.previous_u_lagged = None
        self.previous_w_lagged = None
        self.previous_y_lagged = None

        self.disturbances = None
        self.outputs = None
        self.init_state = None
        self.init_input = None
        self.init_disturbance = None

        self.stage_constraint_func = None
        self.term_constraint_func = None

        self.exp_ineq_constraints = []
        self.exp_eq_constraints = []
        self.n_exp_ineq_constraints = 0
        self.n_exp_eq_constraints = 0

        self.model_type = model_type
        self.mpc_t_step = mpc_t_step

    ####################################################################################################################
    # VARIABLES & PARAMETERS

    def set_opt_vars(self, n_states, n_inputs, n_disturbances, n_outputs):

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_disturbances = n_disturbances
        self.n_outputs = n_outputs
        self.n_stage_vars = self.n_states + self.n_inputs + self.n_disturbances
        self.n_xustage_vars = self.n_states + self.n_inputs
        self.n_total_vars = (self.n_stage_vars * self.n_horizon) + self.n_states
        self.n_optimization_vars = (self.n_states + self.n_inputs) * self.n_horizon

        # indices in z_tau:[u0, x1, u1, s2, ..., xN] corresponding to xk, k neq N
        # a negative index refers to x0
        self.x_stage_indices = np.array([range((k * self.n_xustage_vars) - self.n_states, (k * self.n_xustage_vars))
                                         for k in range(self.n_horizon)])

        # indices in z_tau:[u0, x1, u1, s2, ..., xN] corresponding to uk
        self.u_stage_indices = np.array([range(k * self.n_xustage_vars, (k * self.n_xustage_vars) + self.n_inputs)
                                         for k in range(self.n_horizon)])

        # indices in true_disturbances: w0 w1 ... wn-1 corresponding to wk
        self.w_stage_indices = np.array([range(k * self.n_disturbances, (k + 1) * self.n_disturbances)
                                         for k in range(self.n_horizon)])

        # indices in z:[x0, u0, w0, w1, u1, w1, ..., xN] corresponding to zk, k neq N
        # self.z_stage_indices = np.array([np.hstack([self.x_stage_indices[k], self.u_stage_indices[k],
        #                                            self.w_stage_indices[k]]) for k in range(self.n_horizon)])

        # indices in z:[x0, u0, w0, w1, u1, w1, ..., xN] corresponding to x_lagged, u_lagged, w_lagged
        # where x_lagged: [x_most_lagged ... x_least_lagged, x_now]
        self.x_lagged_indices = np.zeros((self.n_horizon, self.state_lag + 1, self.n_states))
        self.u_lagged_indices = np.zeros((self.n_horizon, self.input_lag + 1, self.n_inputs))
        self.w_lagged_indices = np.zeros((self.n_horizon, self.disturbance_lag + 1, self.n_disturbances))

        for k in range(self.n_horizon):

            for l in range(self.state_lag + 1):
                indices = self.x_stage_indices[k] - ((self.state_lag - l) * self.n_xustage_vars)
                self.x_lagged_indices[k, l, :] = indices

            for l in range(self.input_lag + 1):
                indices = self.u_stage_indices[k] - ((self.input_lag - l) * self.n_xustage_vars)
                self.u_lagged_indices[k, l, :] = indices

            for l in range(self.disturbance_lag + 1):
                indices = self.w_stage_indices[k] - ((self.disturbance_lag - l) * self.n_disturbances)
                self.w_lagged_indices[k, l, :] = indices

        self.x_lagged_indices = np.array(self.x_lagged_indices, dtype='int')
        self.u_lagged_indices = np.array(self.u_lagged_indices, dtype='int')
        self.w_lagged_indices = np.array(self.w_lagged_indices, dtype='int')

        # indices in z:[x0, u0, w0, w1, u1, w1, ..., xN] corresponding to xk+1
        self.next_state_indices = self.x_stage_indices + self.n_xustage_vars
        self.next_state_fullvar_indices = [np.arange((k + 1) * self.n_stage_vars,
                                                     (k + 1) * self.n_stage_vars + self.n_states)
                                           for k in range(self.n_horizon)]
        
        # z_tau: [u0 x1 u1 x2 u2 --- xN] -> z: [u0 x0 u0 w0 x1 u1 w1 --- xN]
        
        # indices in z_tau: [u0 x1 u1 x2 u2 --- xN] corresponding to [u0 x1 u1 x2 u2 ... x_n-1 u_n-1]
        self.xustage_opt_indices = [np.arange(self.n_inputs)] + \
                                   [np.arange(self.n_inputs + k * (self.n_states + self.n_inputs),
                                                         self.n_inputs + (k + 1) * (self.n_states + self.n_inputs))
                                               for k in range(self.n_horizon - 1)]

        self.xterm_opt_indices = np.arange(self.n_optimization_vars - self.n_states, self.n_optimization_vars)
        
        # indices in z:[x0_lagged, u0_lagged, w0_lagged, x1_lagged, u1_lagged, w1_lagged, ..., xN_lagged]
        # corresponding to [u0 x1 u1 x2 u2 ... x_n-1 u_n-1]

        n_lagged_states = self.n_states * (self.state_lag + 1)

        self.xustage_full_indices = [np.arange(self.n_inputs) + n_lagged_states] + \
            [np.hstack([np.arange(self.n_states), np.arange(self.n_inputs) + n_lagged_states])
             for k in range(self.n_horizon - 1)]

    def set_params(self, parameters):
        self.parameters = parameters

    def set_lagged_vars(self, x_lagged, u_lagged, w_lagged):

        self.previous_x_lagged = np.array([None for i in range(self.n_total_vars)])
        self.previous_u_lagged = np.array([None for i in range(self.n_total_vars)])
        self.previous_w_lagged = np.array([None for i in range(self.n_total_vars)])

        for k in range(self.n_horizon):
            idx = 0
            for i in self.x_lagged_indices[k]:
                if np.all(i < 0) and self.previous_x_lagged[i][0] is None:
                    self.previous_x_lagged[i] = x_lagged[idx * self.n_states:(idx + 1) * self.n_states]
                    idx += 1
            idx = 0
            for i in self.u_lagged_indices[k]:
                if np.all(i < 0) and self.previous_u_lagged[i][0] is None:
                    self.previous_u_lagged[i] = u_lagged[idx * self.n_inputs:(idx + 1) * self.n_inputs]
                    idx += 1

            if self.n_disturbances:
                idx = 0
                for i in self.w_lagged_indices[k]:
                    if np.all(i < 0) and self.previous_w_lagged[i][0] is None:
                        self.previous_w_lagged[i] = \
                            w_lagged[idx * self.n_disturbances:(idx + 1) * self.n_disturbances]
                        idx += 1

        return

    def x_stage(self, z_tau, k):
        # if the requested states is included in the optimization variables i.e. x1, x2 ... xN
        if (self.x_stage_indices[k] >= 0).all():
            # return current state
            return z_tau[self.x_stage_indices[k]]
        # else if the requested states are not included in the optimization variables i.e. x0
        else:
            # return most recent lagged state
            return self.previous_x_lagged[self.x_stage_indices[k]]

    def u_stage(self, z_tau, k):
        return z_tau[self.u_stage_indices[k]]

    def w_stage(self, k):
        if self.n_disturbances:
            return self.true_disturbances[self.w_stage_indices[k]]
        else:
            return np.array([])

    def next_states(self, z_tau, k):
        return z_tau[self.next_state_indices[k]]

    def x_lagged(self, z_tau, k):
        x_lagged = []
        # for each lagged state required
        for i in self.x_lagged_indices[k]:
            # if they occurred during this time period of this mpc run
            if (i >= 0).all():
                x_lagged.append(z_tau[i])
            # else if they occurred before this mpc run
            else:
                x_lagged.append(self.previous_x_lagged[i])

        return np.concatenate(x_lagged)

    def u_lagged(self, z_tau, k):
        u_lagged = []
        # for each lagged state required
        for i in self.u_lagged_indices[k]:
            # if they occurred during this time period of this mpc run
            if (i >= 0).all():
                u_lagged.append(z_tau[i])
            # else if they occurred before this mpc run
            else:
                u_lagged.append(self.previous_u_lagged[i])

        return np.concatenate(u_lagged)

    def w_lagged(self, k):
        w_lagged = []
        if self.n_disturbances:
            for i in self.w_lagged_indices[k]:
                # if they occurred during this time period of this mpc run
                if (i >= 0).all():
                    w_lagged.append(self.true_disturbances[i])
                # else if they occurred before this mpc run
                else:
                    w_lagged.append(self.previous_w_lagged[i])
            return np.concatenate(w_lagged)
        else:
            return np.array(w_lagged)

    ####################################################################################################################
    # COST FUNCTION

    def set_cost_funcs(self, stage_cost_func, terminal_cost_func, device_stage_cost_func):
        # single stage
        self.stage_cost_func = stage_cost_func
        self.terminal_cost_func = terminal_cost_func
        self.device_stage_cost_func = device_stage_cost_func

    def set_cost_jacobians(self, stage_cost_jacobian, terminal_cost_jacobian):
        # single function, so jacobian = [gradient]
        self.stage_cost_jacobian = stage_cost_jacobian
        # single function, so jacobian = [gradient]
        self.terminal_cost_jacobian = terminal_cost_jacobian

    def horizon_cost(self, z_tau, k0=0):
        stage_cost = 0
        for k in range(self.n_horizon):
            z_stage = np.hstack([self.x_stage(z_tau, k), self.u_stage(z_tau, k), self.w_stage(k)])
            stage_cost = stage_cost + self.stage_cost_func(z_stage, k0 + k)

        x_term = self.next_states(z_tau, -1)
        term_cost = self.terminal_cost_func(x_term, k0 + self.n_horizon)

        return np.array(float(stage_cost + term_cost))

    def horizon_cost_jacobian(self, z_tau, k0=0):
        # given the current iteration of primal optvars: [u0, x1, u1 ... xN], this function returns the
        # gradient of the horizon cost wrt to the optvars

        optvar_jacobian = np.zeros((1, self.n_optimization_vars))
        
        for k in range(self.n_horizon):
            z_stage = np.hstack([self.x_stage(z_tau, k), self.u_stage(z_tau, k), self.w_stage(k)])

            # indices in the horizon optimization variables z
            optvar_idx = self.xustage_opt_indices[k] 
            fullvar_idx = self.xustage_full_indices[k]

            # gradient of cost as function of current states and inputs
            fullvar_jacobian = self.stage_cost_jacobian(z_stage, k0 + k)
            
            # update the columns of the jacobian corresponding the this stage's variables
            optvar_jacobian[:, optvar_idx] = optvar_jacobian[:, optvar_idx] + fullvar_jacobian[:, fullvar_idx]

        x_term = self.next_states(z_tau, -1)
        fullvar_jacobian = self.terminal_cost_jacobian(x_term, k0 + k)
        optvar_idx = self.xterm_opt_indices
        optvar_jacobian[:, optvar_idx] = optvar_jacobian[:, optvar_idx] + fullvar_jacobian

        return optvar_jacobian

    ####################################################################################################################
    # STATE INEQUALITY CONSTRAINTS

    def set_stage_ineq_constraint_func(self, stage_ineq_constraint_func):

        self.stage_ineq_constraint_f = stage_ineq_constraint_func

    def set_stage_ineq_constraint_jacobian(self, stage_ineq_constraint_jacobian):

        # state inequalities are multiple functions for each time-step (for horizon length > 1),
        # so jacobian = [gradient1; gradient2; ...]
        self.stage_ineq_constraint_jac = stage_ineq_constraint_jacobian

    def stage_ineq_constraint_func(self, z_tau, k0=0):
        cons = []
        for k in range(1, self.n_horizon):
            xu_stage = np.hstack([self.x_stage(z_tau, k), self.u_stage(z_tau, k)])
            cons = np.append(cons, self.stage_ineq_constraint_f(xu_stage, k0 + k, True))

        x_term = self.next_states(z_tau, -1)
        cons = np.append(cons, self.stage_ineq_constraint_f(x_term, k0 + self.n_horizon, False))

        return np.array(cons)

    def stage_ineq_constraint_jacobian(self, z_tau, k0):
        # given the current iteration of primal optvars: [u0, x1, u1 ... xN], this function returns the
        # jacobian of the stage inequalities g wrt to the optvars
        
        optvar_jacobian = np.zeros((self.n_exp_ineq_constraints, self.n_optimization_vars))
        con_start_idx = 0
        for k in range(1, self.n_horizon):
            
            xu_stage = np.hstack([self.x_stage(z_tau, k), self.u_stage(z_tau, k)])
            
            # indices in the horizon optimization variables z
            optvar_idx = self.xustage_opt_indices[k]
            fullvar_idx = self.xustage_full_indices[k]

            fullvar_jacobian = self.stage_ineq_constraint_jac(xu_stage, k0 + k, True)

            con_end_idx = con_start_idx + fullvar_jacobian.shape[0]
            optvar_jacobian[con_start_idx:con_end_idx, optvar_idx] = \
                optvar_jacobian[con_start_idx:con_end_idx, optvar_idx] + fullvar_jacobian[:, fullvar_idx]
            con_start_idx = con_end_idx

        x_term = self.next_states(z_tau, -1)
        fullvar_jacobian = self.stage_ineq_constraint_jac(x_term, k0 + self.n_horizon, False)
        optvar_idx = self.xterm_opt_indices
        optvar_jacobian[con_start_idx:, optvar_idx] = optvar_jacobian[con_start_idx:, optvar_idx] + fullvar_jacobian

        return optvar_jacobian

    ####################################################################################################################
    # NEXT STATE EQUALITY CONSTRAINTS

    def set_next_state_func(self, next_state_func):
        # single-stage
        self.next_state_func = next_state_func

    def disc_calculated_next_states(self, z_tau, k, k0, return_std=False):
        # discrete calculated next states
        z_lagged = np.concatenate([self.x_lagged(z_tau, k), self.u_lagged(z_tau, k), self.w_lagged(k)])
        return self.next_state_func(z_lagged, k0 + k, return_std=return_std)

    def cont_calculated_state_changes(self, z_tau, k, k0):
        # continuous calculated next states
        z_lagged = np.concatenate([self.x_lagged(z_tau, k), self.u_lagged(z_tau, k), self.w_lagged(k)])
        return self.mpc_t_step * self.next_state_func(z_lagged, k0 + k)

    def next_state_constraint_func(self, z_tau, k0=0):

        dynamic_state_cons = []
        if self.model_type == 'continuous':
            for k in range(self.n_horizon):
                dynamic_state_cons.append(z_tau[self.next_state_indices[k]]
                                          - (self.x_stage(z_tau, k) + self.cont_calculated_state_changes(z_tau, k, k0)))

        elif self.model_type == 'discrete':
            for k in range(self.n_horizon):
                dynamic_state_cons.append(z_tau[self.next_state_indices[k]] -
                                          self.disc_calculated_next_states(z_tau, k, k0))
                # TODO replace z_tau with equiv ellipsoid w zero radius for point

        return np.concatenate(dynamic_state_cons)

    def set_next_state_jacobian(self, next_state_jacobian):

        # next states are multiple functions for each time step (for horizon length > 1),
        # so jacobian = [gradient1; gradient2; ...]
        self.next_state_jacobian = next_state_jacobian

    # next_state_constraint_jac
    def next_state_constraint_jacobian(self, z_tau, k0):

        """
        F = lambda z: np.vstack([z[self.next_state_indices[k]][np.newaxis, :].T - self.disc_calculated_next_states(z, k)
                                 for k in range(self.n_horizon)])
        next_states = lambda z: np.vstack([z[self.next_state_indices[k]][np.newaxis, :].T
                                           for k in range(self.n_horizon)])
        calced_next_states = lambda z: np.vstack([self.disc_calculated_next_states(z, k)
                                           for k in range(self.n_horizon)])

        # J_f = jacobian_func(calced_next_states, z)
        # J_x = jacobian_func(next_states, z)
        # J_F = jacobian_func(F, z)
        """

        optvar_jacobian = np.zeros((self.n_horizon * self.n_states, self.n_optimization_vars))

        z_lagged = [np.hstack([self.x_lagged(z_tau, k), self.u_lagged(z_tau, k), self.w_lagged(k)])
                    for k in range(self.n_horizon)]

        # indices in the horizon optimization variables z
        fullvar_idx = self.xustage_full_indices
        optvar_idx = self.xustage_opt_indices

        # indices in z_stage
        # for each next state equation x_k+1 = F(z_lagged)
        for k in range(self.n_horizon):

            for i, d in enumerate(self.next_state_indices[k]):
                optvar_jacobian[k * self.n_states + i, d] = 1

            # df/dxlagged, df/dulagged, df/dwlagged

            next_state_function_fullvar_jacobian = self.next_state_jacobian(z_lagged[k], k=k0 + k)
            
            stage_state_start_idx = k * self.n_states
            stage_state_end_idx = (k + 1) * self.n_states

            optvar_jacobian[stage_state_start_idx:stage_state_end_idx, optvar_idx[k]] = \
                optvar_jacobian[stage_state_start_idx:stage_state_end_idx, optvar_idx[k]] \
                 - next_state_function_fullvar_jacobian[:, fullvar_idx[k]]

        return optvar_jacobian

    ####################################################################################################################
    # IMPLICIT CONSTRAINTS

    def set_imp_bounds(self, stage_bounds, term_bounds):
        self.stage_bounds = stage_bounds  # np.array(list((stage_bounds.values())))
        self.term_bounds = term_bounds  # np.array(list(term_bounds.values()))

    def set_imp_constraints(self, stage_constraint_func, term_constraint_func):
        # take in a single constraint function, which given an array of stage variables will output the value of g(z),
        # which should be >= 0 if z is feasible
        self.stage_constraint_func = stage_constraint_func
        self.term_constraint_func = term_constraint_func

    def implicit_constraint_func(self, z_tau, k0=0):

        imp_stage_cons = np.concatenate([self.stage_constraint_func(np.concatenate(
            [self.x_stage(z_tau, k), self.u_stage(z_tau, k), self.w_stage(k)]), k0 + k)
            for k in range(self.n_horizon)])

        imp_term_cons = self.term_constraint_func(self.next_states(z_tau, -1), k0 + self.n_horizon)

        return np.concatenate([imp_stage_cons, imp_term_cons])

    ####################################################################################################################
    # ALL CONSTRAINTS

    def set_constraints(self):

        self.exp_ineq_constraints.append({'type': 'ineq', 'fun': self.stage_ineq_constraint_func})
        self.n_exp_ineq_constraints += self.stage_ineq_constraint_func(np.zeros(self.n_optimization_vars), 0).shape[0]

        self.exp_eq_constraints.append({'type': 'eq', 'fun': self.next_state_constraint_func})
        self.n_exp_eq_constraints += self.n_horizon * self.n_states

        self.imp_ineq_constraints.append({'type': 'ineq', 'fun': self.implicit_constraint_func})

        self.constraints = self.exp_ineq_constraints + self.exp_eq_constraints + self.imp_ineq_constraints

        return
