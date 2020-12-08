# import casadi as cs
# from casadi.tools import struct_symMX, entry
import numpy as np


# from scipy.linalg import expm
# from helper_functions import gradient_func, jacobian_func


class OCPFunctions:

    def __init__(self):
        pass

    def actual_next_state(self, parameters, t_step, n_states, n_inputs, n_disturbances,
                          state_lag, input_lag, disturbance_lag, is_nonlinear):
        """
        define actual dynamic system model from which measurements can be generated
        :param y: lag * n_y dictionary of past outputs
        :param u: (lag + 1) * n_u array of past and current input
        :param p: n_params array of parameters
        :param w: (lag + 1) * n_y array of pas and current inputs
        :return:
        """
        current_state_indices = list(range(n_states * state_lag,
                                           n_states * (state_lag + 1)))
        current_input_indices = list(range(n_states * (state_lag + 1) + n_inputs * input_lag,
                                           n_states * (state_lag + 1) + n_inputs * (input_lag + 1)))
        current_disturbance_indices = list(range(n_states * (state_lag + 1) + n_inputs * (input_lag + 1)
                                                 + n_disturbances * disturbance_lag,
                                                 n_states * (state_lag + 1) + n_inputs * (input_lag + 1)
                                                 + n_disturbances * (disturbance_lag + 1)))

        # parameters['g'] = 0
        # parameters['eta'] = 0

        def theta_dot(z):
            theta_dot = z[current_state_indices[1]]
            return theta_dot

        def theta_dot_dot(z):
            theta = z[current_state_indices[0]]
            theta_dot = z[current_state_indices[1]]
            u1 = z[current_input_indices[0]]
            sin_theta = np.sin(theta)
            # NOTE: must round result to avoid precision errors when computing jacobian_func, in which a zero difference
            # is not nulled

            res = ((parameters['g'] / parameters['l']) * sin_theta) \
                  - ((parameters['eta'] / (parameters['m'] * parameters['l'] ** 2)) * theta_dot) \
                  + ((1 / (parameters['m'] * parameters['l'] ** 2)) * u1)

            return res

        def cont_func(z):
            state_dot = np.hstack([theta_dot(z), theta_dot_dot(z)])
            return state_dot

        def discretized_nonlinear_func(z):
            # linearize over the neighbourhood of z, then discretize
            # J = jacobian_func(cont_func, z)
            g = parameters['g']
            m = parameters['m']
            eta = parameters['eta']
            l = parameters['l']
            theta_ss = z[current_state_indices[0]]
            A = np.array([[0, 1], [(g / l) * np.cos(theta_ss), -(eta / (m * l ** 2))]])
            B = np.array([[0], [1 / (m * l ** 2)]])
            # A = J[:, current_state_indices]
            # B = J[:, current_input_indices + current_disturbance_indices]

            # A_d = expm(A * t_step)
            A_d = np.eye(A.shape[0]) + (A * t_step)
            B_d = B * t_step
            next_state = (A_d @ z[current_state_indices]) \
                         + B_d @ z[current_input_indices + current_disturbance_indices]

            return next_state

        def discretized_linearized_func(z):

            # linearize entire function
            g = parameters['g']
            m = parameters['m']
            eta = parameters['eta']
            l = parameters['l']
            theta_ss = 0
            A = np.array([[0, 1], [(g / l) * np.cos(theta_ss), -(eta / (m * l ** 2))]])
            B = np.array([[0], [1 / (m * l ** 2)]])

            A_d = np.eye(A.shape[0]) + (A * t_step)
            B_d = B * t_step

            next_state = A_d @ z[current_state_indices] \
                         + B_d @ z[current_input_indices + current_disturbance_indices]

            return next_state

        if is_nonlinear:
            return discretized_nonlinear_func
        else:
            return discretized_linearized_func

    def actual_terminal_cost_func(self, x_term):
        theta = x_term[0]
        # theta_normalized = abs(theta) % (2 * np.pi)
        # cost = min(theta_normalized, abs(2 * np.pi - theta_normalized)) ** 2

        # cost = theta_normalized ** 2

        cost = theta ** 2
        return cost

    def actual_stage_cost_func(self, z_stage):
        """
        :param x: n_y * (n_horizon - 1) array of stage outputs
        :param u: n_u * (n_horizon - 1) array of stage inputs
        :return:
        """
        theta = z_stage[0]
        # theta_normalized = abs(theta) % (2 * np.pi)
        # cost = min(theta_normalized, abs(2 * np.pi - theta_normalized)) ** 2
        # cost = theta_normalized ** 2
        cost = theta ** 2
        return cost

    def stage_bounds(self):
        stage_bounds = {'theta': [-2 * np.pi, 2 * np.pi],
                        'theta_dot': [-np.inf, np.inf],
                        'u1': [-1, 1]}
        return stage_bounds

    def set_bound_constraints(self, bounds, z):
        bound_constraints = np.concatenate([[z[list(bounds.keys()).index(var)] - bound[0],
                                             -z[list(bounds.keys()).index(var)] + bound[1]]
                                            for var, bound in bounds.items()])
        return bound_constraints

    def stage_constraint_func(self, z_stage):

        stage_bounds = self.stage_bounds()
        bound_constraints = self.set_bound_constraints(stage_bounds, z_stage)

        nonbound_constraints = np.array([])

        return np.concatenate([bound_constraints, nonbound_constraints])

    def term_bounds(self):
        term_bounds = {'theta': [-2 * np.pi, 2 * np.pi],
                       'theta_dot': [-np.inf, np.inf]}
        return term_bounds

    def term_constraint_func(self, x_term):

        term_bounds = self.term_bounds()
        bound_constraints = self.set_bound_constraints(term_bounds, x_term)

        nonbound_constraints = np.array([])

        return np.concatenate([bound_constraints, nonbound_constraints])

    def system_variables(self):
        # define system (measurable) states over given lag

        states = ['theta', 'theta_dot']

        inputs = ['u1']

        # define system measurements
        # measurements = states.copy()

        disturbances = []

        # define system parameters

        parameters = {'m': 0.15, 'l': 0.5, 'g': 9.81, 'eta': 0.1}  # {'m': 0.15, 'l': 0.5, 'g': 9.81, 'eta': 0.1}

        numerical_bounds = {'theta': [-3 * np.pi, 3 * np.pi],
                            'theta_dot': [-3 * np.pi, 3 * np.pi],
                            'u1': [-1, 1]}

        return states, inputs, disturbances, parameters, numerical_bounds

    def simulation_variables(self, n_states, n_disturbances, n_simulation_steps, n_horizon):
        init_state = np.array([1, 1])  # [(2 * np.pi) * np.random.uniform(-1, 1), 6.5]
        # if n_disturbances == 0:
        #    disturbances = [[0] for i in range(max(n_simulation_steps, n_horizon))]
        # else:

        disturbances = np.zeros((max(n_simulation_steps, n_horizon), n_disturbances))

        # [[0 for ii in range(n_disturbances)] for i in range(max(n_simulation_steps, n_horizon))]
        # disturbances = list(disturbances.flatten())
        return init_state, disturbances

    def state_prior(self, time_step, n_states, n_inputs, n_disturbances, state_lag, input_lag, disturbance_lag):
        # reduce mass and neglect friction
        parameters = {'m': 0.1, 'l': 0.5, 'g': 9.81, 'eta': 0.0}

        func = self.actual_next_state(parameters, time_step, n_states, n_inputs, n_disturbances, state_lag, input_lag,
                                      disturbance_lag, False)

        return func

    def state_ineq_constraint_func(self, x_stage):
        return np.array([])

    def terminal_cost_jacobian(self, x_term):
        theta = x_term[0]
        grad = [2 * theta, 0]
        return np.vstack([grad])

    def state_ineq_constraint_jacobian(self, x_stages):
        # return vertically stacked gradient_func functions for g_1(x_1), g_2(x_2), ..., g_N(x_N)
        # stage_ineq = lambda x_stage: None
        # return np.vstack([stage_ineq(x_stage) for x_stage in x_stages])
        return np.zeros((0, 2))
