import numpy as np
from helper_functions import gradient_func, jacobian_func


class OCP:

    def __init__(self, n_horizon, state_lag, input_lag, disturbance_lag, t_step, imp_bounds_only=True, model_type='discrete'):

        # horizon length
        self.n_horizon = n_horizon

        # stage cost, terminal cost and dynamic state functions
        self.stage_cost_func = None
        self.terminal_cost_func = None
        self.next_state_funcs = None
        self.state_ineq_func = None

        self.stage_cost_jacobian = None
        self.terminal_cost_jacobian = None
        self.next_state_jacobians = None
        self.state_ineq_jacobian = None

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
        self.n_states = 0
        self.n_inputs = 0
        self.n_disturbances = 0

        # the indices of the mpc primal variables which correspond to the current state at each time step
        self.current_state_indices = None
        # the indices of the mpc primal variables which correspond to the current input at each time step
        self.current_input_indices = None
        # the indices of the mpc primal variables which correspond to the current disturbance at each time step
        self.current_disturbance_indices = None
        # the indices of the mpc primal variables which correspond to the next state at each time step
        self.next_state_indices = None
        # the indices of the mpc primal variables which correspond to the lagged states required at each time step
        self.lagged_state_indices = None
        # the indices of the mpc primal variables which correspond to the lagged inputs required at each time step
        self.lagged_input_indices = None
        # the indices of the mpc primal variables which correspond to the lagged disturbance required at each time step
        self.lagged_disturbance_indices = None

        # indices corresponding to the stage cost input variables from optimization variables z
        self.stage_cost_input_indices = None
        # indices corresponding to the optimization variables from the stage cost jacobian
        self.stage_cost_optvar_indices = None
        # indices corresponding to the state inequality input from optimization variables z
        self.state_ineq_input_indices = None
        # indices corresponding to the optimization variables from the state inequality jacobian
        self.state_ineq_optvar_indices = None
        # indices corresponding to the next state equality input from optimization variables z
        self.next_state_eq_input_indices = None
        # indices corresponding to the optimization variables from the next state equality jacobian
        self.next_state_eq_optvar_indices = None

        self.state_lag = state_lag
        self.input_lag = input_lag
        self.disturbance_lag = disturbance_lag

        # self.current_states = None
        # self.current_inputs = None

        self.previous_lagged_states = None
        self.previous_lagged_inputs = None
        self.previous_lagged_disturbances = None
        self.disturbances = None
        self.init_state = None
        self.init_input = None
        self.init_disturbance = None

        self.stage_constraint_func = None
        self.term_constraint_func = None

        self.exp_ineq_constraints = []
        self.exp_eq_constraints = []
        self.n_exp_ineq_constraints = 0
        self.n_exp_eq_constraints = 0

        # self.n_imp_ineq_constraints = 0
        # self.n_imp_eq_constraints = 0

        self.model_type = model_type
        self.t_step = t_step

    ####################################################################################################################
    # VARIABLES & PARAMETERS

    def set_opt_vars(self, n_states, n_inputs, n_disturbances):

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_disturbances = n_disturbances
        self.n_stage_vars = self.n_states + self.n_inputs
        self.n_total_vars = self.n_stage_vars * self.n_horizon

        self.current_input_indices = np.array([range(k * self.n_stage_vars, (k * self.n_stage_vars) + self.n_inputs)
                                               for k in range(self.n_horizon)])

        self.current_state_indices = np.array([range((k * self.n_stage_vars) - self.n_states, k * self.n_stage_vars)
                                               for k in range(self.n_horizon)])

        self.current_disturbance_indices = np.array([range(k * self.n_disturbances, (k + 1) * self.n_disturbances)
                                                     for k in range(self.n_horizon)])

        self.lagged_state_indices = np.zeros((self.n_horizon, self.state_lag + 1, self.n_states))
        self.lagged_input_indices = np.zeros((self.n_horizon, self.input_lag + 1, self.n_inputs))
        self.lagged_disturbance_indices = np.zeros((self.n_horizon, self.disturbance_lag + 1, self.n_disturbances))

        for k in range(self.n_horizon):

            for l in range(self.state_lag + 1):
                indices = self.current_state_indices[k] - ((self.state_lag - l) * self.n_stage_vars)
                self.lagged_state_indices[k, l, :] = indices

            for l in range(self.input_lag + 1):
                indices = self.current_input_indices[k] - ((self.input_lag - l) * self.n_stage_vars)
                self.lagged_input_indices[k, l, :] = indices

            for l in range(self.disturbance_lag + 1):
                indices = self.current_disturbance_indices[k] - ((self.disturbance_lag - l) * self.n_disturbances)
                self.lagged_disturbance_indices[k, l, :] = indices

        self.lagged_state_indices = np.array(self.lagged_state_indices, dtype='int')
        self.lagged_input_indices = np.array(self.lagged_input_indices, dtype='int')
        self.lagged_disturbance_indices = np.array(self.lagged_disturbance_indices, dtype='int')

        self.next_state_indices = self.current_state_indices + self.n_stage_vars

        self.stage_cost_input_indices = np.hstack([self.current_state_indices, self.current_input_indices])
        self.stage_cost_optvar_indices = [np.where(self.stage_cost_input_indices[k] >= 0)[0] for k in range(self.n_horizon)]
        # remove all variables from before this horizon
        self.stage_cost_input_indices = [self.stage_cost_input_indices[k][self.stage_cost_input_indices[k] >= 0]
                                         for k in range(self.n_horizon)]

        self.state_ineq_input_indices = np.vstack([self.current_state_indices, self.next_state_indices[-1]])
        self.state_ineq_optvar_indices = np.where(self.state_ineq_input_indices.flatten() >= 0)[0]

        # remove all variables from before this horizon
        self.state_ineq_input_indices = self.state_ineq_input_indices[self.state_ineq_input_indices >= 0]
        
        self.next_state_eq_input_indices = np.vstack([
            np.hstack([self.lagged_state_indices[k], self.lagged_input_indices[k]]) for k in range(self.n_horizon)])

        self.next_state_eq_optvar_indices = [np.where(self.next_state_eq_input_indices[k] >= 0)[0]
                                                       for k in range(self.n_horizon)]

        # remove all variables from before this horizon
        self.next_state_eq_input_indices = [self.next_state_eq_input_indices[k][self.next_state_eq_input_indices[k] >= 0]
                                            for k in range(self.n_horizon)]

        # remove disturbance indices from input indices
        for k in range(self.n_horizon):
            for i in self.next_state_eq_input_indices[k]:
                if i in self.current_disturbance_indices or i < 0:
                    self.next_state_eq_input_indices[k].remove(i)

    def set_params(self, parameters):
        self.parameters = parameters

    def set_lagged_vars(self, lagged_states, lagged_inputs, lagged_disturbances):

        self.previous_lagged_states = np.array([None for i in range(self.n_total_vars)])
        self.previous_lagged_inputs = np.array([None for i in range(self.n_total_vars)])
        self.previous_lagged_disturbances = np.array([None for i in range(self.n_total_vars)])

        for k in range(self.n_horizon):
            idx = 0
            for i in self.lagged_state_indices[k]:

                if np.all(i < 0) and self.previous_lagged_states[i][0] is None:
                    self.previous_lagged_states[i] = lagged_states[idx * self.n_states:(idx + 1) * self.n_states]
                    idx += 1
            idx = 0
            for i in self.lagged_input_indices[k]:
                if np.all(i < 0) and self.previous_lagged_inputs[i][0] is None:
                    self.previous_lagged_inputs[i] = lagged_inputs[idx * self.n_inputs:(idx + 1) * self.n_inputs]
                    idx += 1

            if self.n_disturbances:
                idx = 0
                for i in self.lagged_disturbance_indices[k]:
                    if np.all(i < 0) and self.previous_lagged_disturbances[i][0] is None:
                        self.previous_lagged_disturbances[i] = \
                            lagged_disturbances[idx * self.n_disturbances:(idx + 1) * self.n_disturbances]
                        idx += 1

        return

    def current_states(self, z, k):
        # if the requested states is included in the optimization variables i.e. x1, x2 ... xN
        if (self.current_state_indices[k] >= 0).all():
            # return current state
            return z[self.current_state_indices[k]]
        # else if the requested states are not included in the optimization variables i.e. x0
        else:
            # return most recent lagged state
            return self.previous_lagged_states[self.current_state_indices[k]]

    def current_inputs(self, z, k):
        if (self.current_input_indices[k] >= 0).all():
            return z[self.current_input_indices[k]]
        else:
            return self.previous_lagged_inputs[self.current_state_indices[k]]

    def current_disturbances(self, z, k):
        if self.n_disturbances:
            if (self.current_disturbance_indices[k] >= 0).all():
                return z[self.current_disturbance_indices[k]]
            else:
                return self.previous_lagged_disturbances[self.current_disturbance_indices[k]]
        else:
            return np.array([])

    def next_states(self, z, k):
        return z[self.next_state_indices[k]]

    def lagged_states(self, z, k):
        lagged_states = []
        # for each lagged state required
        for i in self.lagged_state_indices[k]:
            # if they occurred during this time period of this mpc run
            if (i >= 0).all():
                lagged_states.append(z[i])
            # else if they occurred before this mpc run
            else:
                lagged_states.append(self.previous_lagged_states[i])

        return np.concatenate(lagged_states)

    def lagged_inputs(self, z, k):
        lagged_inputs = []
        # for each lagged state required
        for i in self.lagged_input_indices[k]:
            # if they occurred during this time period of this mpc run
            if (i >= 0).all():
                lagged_inputs.append(z[i])
            # else if they occurred before this mpc run
            else:
                lagged_inputs.append(self.previous_lagged_inputs[i])

        return np.concatenate(lagged_inputs)

    def lagged_disturbances(self, z, k):
        lagged_disturbances = []
        if self.n_disturbances:
            for i in self.lagged_disturbance_indices[k]:
                # if they occurred during this time period of this mpc run
                if (i >= 0).all():
                    lagged_disturbances.append(z[i])
                # else if they occurred before this mpc run
                else:
                    lagged_disturbances.append(self.previous_lagged_disturbances[i])
            return np.concatenate(lagged_disturbances)
        else:
            return np.array(lagged_disturbances)

    ####################################################################################################################
    # COST FUNCTION

    def set_cost_funcs(self, stage_cost_func, terminal_cost_func):
        # single stage
        self.stage_cost_func = stage_cost_func
        self.terminal_cost_func = terminal_cost_func

    def set_cost_jacobians(self, stage_cost_jacobian, terminal_cost_jacobian):
        # single function, so jacobian = [gradient]
        self.stage_cost_jacobian = stage_cost_jacobian
        # single function, so jacobian = [gradient]
        self.terminal_cost_jacobian = terminal_cost_jacobian

    def horizon_cost(self, z):
        stage_cost = 0
        for k in range(self.n_horizon):
            z_stage = np.hstack([self.current_states(z, k), self.current_inputs(z, k)])
            stage_cost += self.stage_cost_func(z_stage)

        x_term = self.next_states(z, -1)
        term_cost = self.terminal_cost_func(x_term)

        return float(stage_cost + term_cost)

    def horizon_cost_jacobian(self, z):

        jacobian = np.zeros((1, self.n_total_vars))
        for k in range(self.n_horizon):
            z_stage = np.hstack([self.current_states(z, k), self.current_inputs(z, k)])

            # indices in the horizon optimization variables z
            input_idx = self.stage_cost_input_indices[k]
            optvar_idx = self.stage_cost_optvar_indices[k]

            # gradient of cost as function of current states and inputs
            # TODO DOUBLE CHECK GP GRADIENT FUNCTION, PRODUCING NONZERO VALUES
            # TODO DOULBLE CHECK DYNAMIC EQUATIONS, PENDULUM SHOULD NOT BE SWINGING ON ITS OWN
            #  should be theta * [0, 2, 0, 0, 2, 0, 0, 2, 0]
            jac = self.stage_cost_jacobian(z_stage)

            jacobian[:, input_idx] = jacobian[:, input_idx] + jac[:, optvar_idx]

        x_term = self.next_states(z, -1)
        jac = self.terminal_cost_jacobian(x_term)
        input_idx = self.next_state_indices[-1]
        jacobian[:, input_idx] = jacobian[:, input_idx] + jac

        return jacobian

    ####################################################################################################################
    # STATE INEQUALITY CONSTRAINTS

    def set_state_ineq_func(self, state_ineq_func):
        self.state_ineq_func = state_ineq_func

    def set_state_ineq_jacobian(self, state_ineq_jacobian):

        # state inequalities are multiple functions for each time-step (for horizon length > 1),
        # so jacobian = [gradient1; gradient2; ...]
        self.state_ineq_jacobian = state_ineq_jacobian

    def state_ineq_constraint_jacobian(self, z):

        # return np.vstack([con['jac'](z) for con in self.exp_ineq_constraints])

        jacobian = np.zeros((self.n_exp_ineq_constraints, self.n_total_vars))

        # for k in range(self.n_horizon):

        x_stages = [self.current_states(z, k) for k in range(self.n_horizon)] + self.next_states(z, -1)

        # indices in the horizon optimization variables z
        input_idx = self.state_ineq_input_indices

        optvar_idx = self.state_ineq_optvar_indices

        # this jacobian is a function of x_stages variables only
        jac = self.state_ineq_jacobian(x_stages)

        jacobian[:, input_idx] = jacobian[:, input_idx] + jac[:, optvar_idx]

        return jacobian

    ####################################################################################################################
    # NEXT STATE EQUALITY CONSTRAINTS

    def set_next_state_funcs(self, next_state_funcs):
        # single-stage
        self.next_state_funcs = next_state_funcs

    def disc_calculated_next_states(self, z, k):
        # discrete calculated next states
        return np.hstack([self.next_state_funcs[d](
            np.concatenate([self.lagged_states(z, k), self.lagged_inputs(z, k),
                            self.lagged_disturbances(z, k)])) for d in range(self.n_states)])

    def cont_calculated_state_changes(self, z, k):
        # continuous calculated next states
        return np.hstack([self.t_step * self.next_state_funcs[d](
            np.concatenate([self.lagged_states(z, k), self.lagged_inputs(z, k),
                            self.lagged_disturbances(z, k)])) for d in range(self.n_states)])

    def next_state_constraint_func(self, z):

        dynamic_state_cons = []
        if self.model_type == 'continuous':
            for k in range(self.n_horizon):
                dynamic_state_cons.append(z[self.next_state_indices[k]]
                                          - (self.current_states(z, k) + self.cont_calculated_state_changes(z, k)))

        elif self.model_type == 'discrete':
            for k in range(self.n_horizon):
                dynamic_state_cons.append(z[self.next_state_indices[k]] - self.disc_calculated_next_states(z, k))

        return np.concatenate(dynamic_state_cons)

    def set_next_state_jacobians(self, next_state_jacobians):

        # next states are multiple functions for each time step (for horizon length > 1),
        # so jacobian = [gradient1; gradient2; ...]
        self.next_state_jacobians = next_state_jacobians

    # next_state_constraint_jac
    def next_state_constraint_jacobian(self, z):

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

        # return np.vstack([con['jac'](z) for con in self.exp_eq_constraints])
        # return np.vstack([self.next_state_jacobian(np.hstack([self.lagged_states(z, k),
        #                 self.lagged_inputs(z, k),
        #                 self.lagged_disturbances(z, k)])) for k in range(self.n_horizon)])

        jacobian = np.zeros((self.n_horizon * self.n_states, self.n_total_vars))

        z_lagged = [np.hstack([self.lagged_states(z, k), self.lagged_inputs(z, k), self.lagged_disturbances(z, k)])
                    for k in range(self.n_horizon)]

        # indices in the horizon optimization variables z
        input_idx = self.next_state_eq_input_indices
        optvar_idx = self.next_state_eq_optvar_indices

        # indices in z_stage

        for k in range(self.n_horizon):

            jac_x = np.zeros((self.n_states, self.n_total_vars))
            for i, d in enumerate(self.next_state_indices[k]):
                jac_x[i, d] = 1

            stage_state_start_idx = k * self.n_states
            stage_state_end_idx = (k + 1) * self.n_states

            jacobian[stage_state_start_idx:stage_state_end_idx] = \
                jacobian[stage_state_start_idx:stage_state_end_idx] + jac_x

            # jacobian_func(self.next_state_func, z_lagged[k])
            jac_f = np.vstack([self.next_state_jacobians[d](z_lagged[k]) for d in range(self.n_states)])

            jacobian[stage_state_start_idx:stage_state_end_idx, input_idx[k]] = \
                jacobian[stage_state_start_idx:stage_state_end_idx, input_idx[k]] - jac_f[:, optvar_idx[k]]

        return jacobian

    ####################################################################################################################
    # IMPLICIT CONSTRAINTS

    def set_imp_bounds(self, stage_bounds, term_bounds):
        self.stage_bounds = np.array(list((stage_bounds.values())))
        self.term_bounds = np.array(list(term_bounds.values()))

    def set_imp_constraints(self, stage_constraint_func, term_constraint_func):
        # take in a single constraint function, which given an array of stage variables will output the value of g(z),
        # which should be >= 0 if z is feasible
        self.stage_constraint_func = stage_constraint_func
        self.term_constraint_func = term_constraint_func

    def implicit_constraint_func(self, z):

        imp_stage_cons = np.concatenate([self.stage_constraint_func(np.concatenate(
            [self.current_states(z, k), self.current_inputs(z, k), self.current_disturbances(z, k)]))
            for k in range(self.n_horizon)])

        imp_term_cons = self.term_constraint_func(self.next_states(z, -1))

        return np.concatenate([imp_stage_cons, imp_term_cons])

    ####################################################################################################################
    # ALL CONSTRAINTS

    def set_constraints(self):

        # define x0
        # self.init_state = init_state

        # define w_0, w_1 ... w_N-1
        # self.disturbances = disturbances

        # define g_1, g_2 ... g_N

        self.exp_ineq_constraints.append({'type': 'ineq', 'fun': self.state_ineq_func})
        self.n_exp_ineq_constraints += self.state_ineq_func(np.zeros(self.n_total_vars)).shape[0]

        self.exp_eq_constraints.append({'type': 'eq', 'fun': self.next_state_constraint_func})
        self.n_exp_eq_constraints += self.n_horizon * self.n_states

        self.imp_ineq_constraints.append({'type': 'ineq', 'fun': self.implicit_constraint_func})

        self.constraints = self.exp_ineq_constraints + self.exp_eq_constraints + self.imp_ineq_constraints

        return

