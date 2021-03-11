import numpy as np
from helper_functions import jacobian_func


class Model:
    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps):

        self.states = None
        self.inputs = None
        self.stage_costs = None
        self.outputs = None

        self.prior_parameters = None
        self.numerical_bounds = None
        self.numerical_gaussian = None

        self.init_state = None
        self.ref_state = None
        self.disturbances = None
        self.outputs = None
        self.parameters = None

        self.x_stage_indices = None
        self.u_stage_indices = None
        self.xu_stage_indices = None
        self.w_stage_indices = None

        self.t_step = t_step
        self.is_nonlinear = None
        self.is_discrete = None

        self.n_states = None
        self.n_inputs = None
        self.n_disturbances = None
        self.n_outputs = None

        self.state_lag = state_lag
        self.input_lag = input_lag
        self.disturbance_lag = disturbance_lag
        self.output_lag = output_lag

        self.n_horizon = n_horizon
        self.n_simulation_steps = n_simulation_steps
        self.simulation_step = 0

        self.disturbance_train = None

    def set_indices(self):

        self.n_states = len(self.states)
        self.n_inputs = len(self.inputs)
        self.n_disturbances = len(self.disturbances)
        self.n_outputs = len(self.outputs)

        # current states are at the end of the given vector
        self.x_stage_indices = list(range(self.n_states * self.state_lag,
                                          self.n_states * (self.state_lag + 1)))

        self.u_stage_indices = list(range(self.n_states * (self.state_lag + 1)
                                          + self.n_inputs * self.input_lag,
                                          self.n_states * (self.state_lag + 1)
                                          + self.n_inputs * (self.input_lag + 1)))
        self.xu_stage_indices = np.hstack([self.x_stage_indices, self.u_stage_indices])

        self.w_stage_indices = list(range(self.n_states * (self.state_lag + 1)
                                          + self.n_inputs * (self.input_lag + 1)
                                          + self.n_disturbances * self.disturbance_lag,
                                          self.n_states * (self.state_lag + 1)
                                          + self.n_inputs * (self.input_lag + 1)
                                          + self.n_disturbances * (self.disturbance_lag + 1)))

    def system_variables(self):
        pass

    def cont_func(self, z_lagged):
        # z_lagged = [x_lagged, u_lagged, w_lagged]
        pass

    def true_discrete_func(self, z_lagged):
        # z_lagged = [x_lagged, u_lagged, w_lagged]
        pass

    def linear_discretized_func(self, z_lagged):
        # z_lagged = [x_lagged, u_lagged, w_lagged]
        pass

    def euler_discretized_func(self, z_lagged):
        # z_lagged = [x_lagged, u_lagged, w_lagged]
        # linearize over the neighbourhood of z, then discretize

        next_state = z_lagged[self.x_stage_indices] + self.t_step * (self.cont_func(z_lagged))

        return next_state

    def rk_discretized_func(self, z_lagged, cont_func=None):

        # z_lagged = [x_lagged, u_lagged, w_lagged]

        if cont_func is None:
            cont_func = self.true_next_state_cont_func

        k1 = cont_func(z_lagged)

        z_delta = np.array(z_lagged)
        z_delta[self.x_stage_indices] = z_delta[self.x_stage_indices] + (self.t_step * (k1 / 2))
        k2 = cont_func(z_delta)

        z_delta = np.array(z_lagged)
        z_delta[self.x_stage_indices] = z_delta[self.x_stage_indices] + (self.t_step * (k2 / 2))
        k3 = cont_func(z_delta)

        z_delta = np.array(z_lagged)
        z_delta[self.x_stage_indices] = z_delta[self.x_stage_indices] + (self.t_step * k3)
        k4 = cont_func(z_delta)

        next_state = z_lagged[self.x_stage_indices] + ((1 / 6) * (k1 + (2 * k2) + (2 * k3) + k4))

        return next_state

    def true_next_state_func(self, z_lagged):

        if self.is_discrete:
            return self.true_next_state_discrete_func(z_lagged)
        elif self.is_nonlinear:
            return self.rk_discretized_func(z_lagged)
            # return self.euler_discretized_func(z_lagged)
        else:
            return self.linear_discretized_func(z_lagged)

    def stage_bounds(self, z_stage):
        pass

    def set_bound_constraints(self, state_bounds=None, input_bounds=None, x=None, u=None):
        if state_bounds is not None:
            state_bound_constraints = [[x[idx] - bound[0], -x[idx] + bound[1]]
                                       for idx, bound in enumerate(state_bounds)]
        else:
            state_bound_constraints = []

        if input_bounds is not None:
            input_bound_constraints = [[u[idx] - bound[0], -u[idx] + bound[1]]
                                       for idx, bound in enumerate(input_bounds)]
        else:
            input_bound_constraints = []

        bound_constraints = np.concatenate(state_bound_constraints + input_bound_constraints)
        return bound_constraints

    def stage_constraint_func(self, z_stage, k):
        self.set_simulation_step(k)
        stage_bounds = self.stage_bounds(z_stage)
        state_bounds = stage_bounds[:self.n_states]
        input_bounds = stage_bounds[self.n_states:]

        bound_constraints = self.set_bound_constraints(state_bounds=state_bounds, input_bounds=input_bounds,
                                                       x=z_stage[self.x_stage_indices],
                                                       u=z_stage[self.u_stage_indices])

        nonbound_constraints = np.array([])

        return np.concatenate([bound_constraints, nonbound_constraints])

    def term_constraint_func(self, x_term, k):
        self.set_simulation_step(k)
        state_bounds = self.term_bounds(x_term)
        bound_constraints = self.set_bound_constraints(state_bounds=state_bounds, x=x_term)

        nonbound_constraints = np.array([])

        return np.concatenate([bound_constraints, nonbound_constraints])

    def set_simulation_step(self, k):
        self.simulation_step = k


class TCLModel(Model):
    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps, idx=0):
        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps)

        self.is_nonlinear = True
        self.is_discrete = True

        self.idx = idx
        self.name = 'TCL'

        self.system_variables()
        self.set_indices()

        self.set_ref_state()
        self.set_init_state()

    def system_variables(self):
        # define system (measurable) states over given lag

        self.states = [f'T_zone_{self.idx}']

        self.inputs = [f'P_zone_{self.idx}']  # [f'm_dot_a_{self.idx}', 'T_a']

        self.disturbances = ['T_outside']  # f'Q_s_{self.idx}', f'Q_i_{self.idx}']

        self.outputs = []

        # define system parameters
        T_zone_ref = [24., 23.5, 24.5, 23.5, 24.5]
        T_zone_min = [22., 21.5, 22.5, 21, 23]
        T_zone_max = [28, 28.5, 28, 28.5, 27.5]

        self.original_parameters = {'T_zone_ref': T_zone_ref[self.idx]
                                                  * np.ones(self.n_simulation_steps + self.n_horizon),
                                    'P_zone_max': np.zeros(self.n_simulation_steps + self.n_horizon),
                                    'P_zone_min': -48 * np.ones(self.n_simulation_steps + self.n_horizon),
                                    'T_zone_min': T_zone_min[self.idx]
                                                  * np.ones(int(self.n_simulation_steps + self.n_horizon)),
                                    'T_zone_max': T_zone_max[self.idx]
                                                  * np.ones(int(self.n_simulation_steps + self.n_horizon))}

        # [-48 for i in range(int(self.n_simulation_steps / 4))]
        # + [-24 for i in range(int(self.n_simulation_steps / 4))]
        # + [-5 for i in range(int(self.n_simulation_steps / 4))]
        # + [-10 for i in range(int(self.n_simulation_steps / 4))]
        # + [-5 for i in range(self.n_horizon + self.n_simulation_steps
        #                      - 4 * int(self.n_simulation_steps / 4))],

        self.parameters = dict(self.original_parameters)

        self.numerical_bounds = {f'T_zone_{self.idx}': [T_zone_min[self.idx], T_zone_max[self.idx]],
                                 f'P_zone_{self.idx}': [-48, 0],
                                 # f'm_dot_a_{self.idx}': [0.2, 2.2],
                                 #  'T_a': [10, 16],
                                 'T_outside': [22, 32.5]}
        # f'Q_s_{self.idx}': [0, 6.9],
        # f'Q_i_{self.idx}': [0.17, 5.04]}

        self.numerical_gaussian = {f'T_zone_{self.idx}': {'mean': 24.2, 'std': 1.05},
                                   f'P_zone_{self.idx}': {'mean': -0.81 * 12, 'std': 0.262 * 12},
                                   # f'm_dot_a_{self.idx}': {'mean': 0.29, 'std': 0.13},
                                   # 'T_a': {'mean': 13, 'std': 0.9},
                                   'T_outside': {'mean': 28.8, 'std': 3.7}}
        # f'Q_s_{self.idx}': {'mean': 1.5, 'std': 2},
        # f'Q_i_{self.idx}': {'mean': 0.79, 'std': 0.71}}

    def set_ref_state(self, ref_state=None):
        self.ref_state = np.array([self.parameters['T_zone_ref'][0]]) if ref_state is None else ref_state

    def set_init_state(self, init_state=None):
        self.init_state = np.array(self.ref_state) if init_state is None else init_state

    def stage_bounds(self, z_stage):

        stage_bounds = {f'T_zone_{self.idx}': [-np.inf, np.inf],#[self.parameters['T_zone_min'][self.simulation_step],
                                               #self.parameters['T_zone_max'][self.simulation_step]],
                        f'P_zone_{self.idx}': [self.parameters['P_zone_min'][self.simulation_step],
                                               self.parameters['P_zone_max'][self.simulation_step]]}
        return np.vstack(list(stage_bounds.values()))

    def term_bounds(self, x_term):
        term_bounds = {f'T_zone_{self.idx}':[-np.inf, np.inf]}
                          # [self.parameters['T_zone_min'][self.simulation_step],
                           # self.parameters['T_zone_max'][self.simulation_step]]}
        return np.vstack(list(term_bounds.values()))

    def next_state_prior_func(self, z_lagged):
        # TODO
        # T_zone = z_lagged[self.x_stage_indices[0]]
        # return np.array([T_zone])
        return np.array([0])

    def linearized_next_state_prior_func(self, z_lagged, z_lagged_bar):
        return np.array([0])

    def linearized_next_state_prior_jacobian(self, z_lagged, z_lagged_bar):
        return np.array([0])

    def next_state_prior_jacobian(self, z_lagged):
        jac = np.zeros((self.n_states, z_lagged.shape[0]))
        # T_zone_idx = self.x_stage_indices[0]
        # jac[0, T_zone_idx] = 1
        return jac

    def stage_cost_prior_func(self, z_stage):
        return np.array([0])

    def stage_cost_prior_jacobian(self, z_stage):
        jac = np.zeros((1, z_stage.shape[0]))
        return jac

    def P_out_stage(self, z_stage):
        # m_dot_a = z_stage[self.u_stage_indices[0]]
        # T_outside = z_stage[self.w_stage_indices[0]]
        # T_a = z_stage[self.u_stage_indices[1]]
        # P_fan = 0.0076 * m_dot_a ** 3 + 4.8865
        # P_chiller = z_stage[self.u_stage_indices[0]] * (T_outside - T_a) if T_outside > T_a else 0
        # return - (P_fan + P_chiller)
        P_zone = z_stage[self.u_stage_indices[0]]
        return np.array([P_zone])

    def P_out_stage_jacobian(self, z_stage):
        # m_dot_a_idx = self.u_stage_indices[0]
        # m_dot_a = z_stage[m_dot_a_idx]
        # T_outside_idx = self.w_stage_indices[0]
        # T_outside = z_stage[T_outside_idx]
        # T_a_idx = self.u_stage_indices[1]
        # T_a = z_stage[T_a_idx]
        #
        # jac = np.zeros((1, z_stage.shape[0]))
        #
        # jac[0, m_dot_a_idx] = 0.0076 * 3 * m_dot_a ** 2 + ((T_outside - T_a) if T_outside > T_a else 0)
        # jac[0, T_a_idx] = -m_dot_a if T_outside > T_a else 0
        # jac[0, T_outside_idx] = m_dot_a if T_outside > T_a else 0
        # return -jac
        return jacobian_func(self.P_out_stage, z_stage)

    def P_out_term(self, x_term):
        return 0

    def P_out_term_jacobian(self, x_term):
        jac = np.zeros((1, x_term.shape[0]))
        return jac

    def true_stage_cost_func(self, z_stage, return_std=False):
        T_zone = z_stage[self.x_stage_indices[0]]
        res = np.array([(T_zone - self.parameters['T_zone_ref'][self.simulation_step]) ** 2])
        if return_std:
            return res, [0]
        else:
            return res

    def true_stage_cost_jacobian(self, z_stage):
        T_zone_idx = self.x_stage_indices[0]
        T_zone = z_stage[T_zone_idx]
        jac = np.zeros((1, z_stage.shape[0]))
        jac[0, T_zone_idx] = 2 * (T_zone - self.parameters['T_zone_ref'][self.simulation_step])
        return jac

    def terminal_cost_func(self, x_term):
        T_zone = x_term[0]
        return (T_zone - self.parameters['T_zone_ref'][self.simulation_step]) ** 2

    def terminal_cost_jacobian(self, x_term):
        T_zone_idx = self.x_stage_indices[0]
        T_zone = x_term[T_zone_idx]
        jac = np.zeros((1, x_term.shape[0]))
        jac[0, T_zone_idx] = 2 * (T_zone - self.parameters['T_zone_ref'][self.simulation_step])
        return jac

    def stage_ineq_constraint_func(self, xu_stage):
        T_zone = xu_stage[self.x_stage_indices[0]]
        g = [T_zone - self.parameters['T_zone_max'][self.simulation_step],
             -T_zone + self.parameters['T_zone_min'][self.simulation_step]]
        # g = []
        return np.array([g])

    def stage_ineq_constraint_jacobian(self, xu_stage):
        T_zone_idx = self.x_stage_indices[0]
        jac = np.zeros((2, xu_stage.shape[0]))
        jac[0, T_zone_idx] = 1
        jac[1, T_zone_idx] = -1
        return jac


class TCLModel2(TCLModel):
    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps, idx=0):
        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps, idx)

        self.system_variables()
        self.set_indices()

        self.set_ref_state()
        self.set_init_state()

    def true_next_state_discrete_func(self, z_lagged):
        delta_t = self.parameters['mpc_t_step']
        C = self.parameters['C']
        R = self.parameters['R']
        T_zone = z_lagged[self.x_stage_indices[0]]
        P_zone = z_lagged[self.u_stage_indices[0]]
        T_outside = z_lagged[self.w_stage_indices[0]]
        T_gain = R * P_zone

        a = np.exp(-delta_t / (C * R))
        T_next = a * T_zone + (1 - a) * (T_outside - T_gain)

        return np.array([T_next])

    def system_variables(self):
        # define system (measurable) states over given lag

        self.states = [f'T_zone_{self.idx}']

        self.inputs = [f'P_zone_{self.idx}']  # [f'm_dot_a_{self.idx}', 'T_a']

        self.disturbances = ['T_outside']  # f'Q_s_{self.idx}', f'Q_i_{self.idx}']

        self.outputs = []

        # define system parameters
        T_zone_ref = [21., 22, 23, 24, 25]
        T_zone_min = [20, 21, 22, 23, 24]
        T_zone_max = [25, 26, 27, 28, 29]
        C = [12, 8, 10, 9, 11]
        R = [2, 1.75, 1.5, 2.5, 2.25]

        self.original_parameters = {'T_zone_ref': T_zone_ref[self.idx]
                                                  * np.ones(self.n_simulation_steps + self.n_horizon),
                                    'P_zone_min': 8 * np.ones(self.n_simulation_steps + self.n_horizon),
                                    'P_zone_max': 24 * np.ones(self.n_simulation_steps + self.n_horizon),
                                    'T_zone_min': T_zone_min[self.idx]
                                                  * np.ones(int(self.n_simulation_steps + self.n_horizon)),
                                    'T_zone_max': T_zone_max[self.idx]
                                                  * np.ones(int(self.n_simulation_steps + self.n_horizon)),
                                    'R': R[self.idx], 'C': C[self.idx], 'mpc_t_step': self.t_step}

        self.parameters = dict(self.original_parameters)

        self.numerical_bounds = {f'T_zone_{self.idx}': [T_zone_min[self.idx], T_zone_max[self.idx]],
                                 f'P_zone_{self.idx}': [6, 26],
                                 'T_outside': [22, 33],}

class BatteryModel(Model):
    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps, idx=1):
        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps)
        self.is_nonlinear = True
        self.is_discrete = True
        self.name = 'Battery'

        self.idx = idx
        self.system_variables()
        self.set_indices()
        self.set_init_state()

    def system_variables(self):
        # define system (measurable) states over given lag

        self.states = [f'soc_{self.idx}']

        self.inputs = [f'E_out_{self.idx}']

        self.disturbances = [f'P_ref_{self.idx}']

        self.outputs = []

        # define system parameters

        self.original_parameters = {'P_ref': [5000 for i in range(int(self.n_simulation_steps / 4))]
                                             + [-5000 for i in range(int(self.n_simulation_steps / 4))]
                                             + [5000 for i in range(int(self.n_simulation_steps / 4))]
                                             + [-5000 for i in range(int(self.n_simulation_steps / 4))]
                                             + [0 for i in range(self.n_horizon + self.n_simulation_steps
                                                                 - 4 * int(self.n_simulation_steps / 4))],
                                    'E_out_min': -8593 * (5 / 60) * np.ones(self.n_simulation_steps + self.n_horizon),
                                    'E_out_max': 9170 * (5 / 60) * np.ones(self.n_simulation_steps + self.n_horizon),
                                    'soc_min': np.zeros(self.n_simulation_steps + self.n_horizon),
                                    'soc_max': np.ones(self.n_simulation_steps + self.n_horizon)}

        self.parameters = dict(self.original_parameters)

        self.numerical_bounds = {  # f'E_{self.idx}': [4705, 6472], #[0, 1],
            # f'E_out_{self.idx}': [-10715, 26530],
            f'soc_{self.idx}': [0.45, 0.6],
            f'E_out_{self.idx}': [-9000 * (5 / 60), 9000 * (5 / 60)],  # [-8593 * (5 / 60), 9170 * (5 / 60)],
            f'P_ref_{self.idx}': [-9000, 9000]}  # [-8593, 9170]}

        self.numerical_gaussian = {f'soc_{self.idx}': {'mean': 0.55, 'std': 0.011},
                                   f'E_out_{self.idx}': {'mean': -61, 'std': 918},
                                   f'P_ref_{self.idx}': {'mean': -61, 'std': 918}}

    def set_init_state(self):
        self.init_state = np.array([0.5])

    def stage_bounds(self, z_stage):
        E_out = z_stage[self.u_stage_indices[0]]
        stage_bounds = {  # f'E_{self.idx}': [4705, 6472],
            # f'E_out_{self.idx}': [-10715, 26530],
            f'soc_{self.idx}': [0, 1],
            f'E_out_{self.idx}': [0, self.parameters['E_out_max'][self.simulation_step]] if E_out >= 0
            else [self.parameters['E_out_min'][self.simulation_step], 0]}

        return np.vstack(list(stage_bounds.values()))

    def term_bounds(self, x_term):
        term_bounds = {  # f'E_{self.idx}': [4705, 6472]
            'soc': [0, 1]
        }
        return np.vstack(list(term_bounds.values()))

    def next_state_prior_func(self, z_lagged):
        soc_prior = z_lagged[self.x_stage_indices[0]]
        return np.array([soc_prior])
        # return np.array([0])

    def next_state_prior_jacobian(self, z_lagged):
        jac = np.zeros((self.n_states, z_lagged.shape[0]))
        soc_idx = self.x_stage_indices[0]
        jac[0, soc_idx] = 1
        return jac

    def true_next_state_discrete_func(self, z_lagged):
        soc = z_lagged[self.x_stage_indices[0]]
        E_out = z_lagged[self.u_stage_indices[0]]
        x = np.hstack([[1], soc, E_out])
        coeffs = [-3.46320610e-01, 2.43797077e+00, -3.23409152e-04, -1.49794845e+00, 4.54199086e-04, -2.94533953e-08]
        powers = [[2, 0, 0],
                  [1, 1, 0],
                  [1, 0, 1],
                  [0, 2, 0],
                  [0, 1, 1],
                  [0, 0, 2]]
        soc_next = np.sum([c * (x ** p).prod() for c, p in zip(coeffs, powers)])
        return np.array([soc_next])

    def stage_cost_prior_func(self, z_stage):
        return np.array([0])

    def stage_cost_prior_jacobian(self, z_stage):
        jac = np.zeros((1, z_stage.shape[0]))
        return jac

    def P_out_stage(self, z_stage):
        E_out = z_stage[self.u_stage_indices[0]]
        return E_out / (5 / 60)

    def P_out_stage_jacobian(self, z_stage):
        P_idx = self.u_stage_indices[0]
        jac = np.zeros((1, z_stage.shape[0]))
        jac[0, P_idx] = 1 / (5 / 60)
        return jac

    def P_out_term(self, x_term):
        return 0

    def P_out_term_jacobian(self, x_term):
        jac = np.zeros((1, x_term.shape[0]))
        return jac

    def true_stage_cost_func(self, z_stage):
        P_out = z_stage[self.u_stage_indices[0]] / (5 / 60)
        P_ref = z_stage[self.w_stage_indices[0]]
        return np.array([(P_out - P_ref) ** 2])

    def true_stage_cost_jacobian(self, z_stage):
        E_out_idx = self.u_stage_indices[0]
        E_out = z_stage[E_out_idx]
        P_out = E_out / (5 / 60)
        P_ref_idx = self.w_stage_indices[0]
        P_ref = z_stage[P_ref_idx]
        jac = np.zeros((1, z_stage.shape[0]))
        jac[0, E_out_idx] = 2 / (5 / 60) * (P_out - P_ref)
        jac[0, P_ref_idx] = -2 * (P_out - P_ref)
        return jac

    def terminal_cost_func(self, x_term):
        return 0

    def terminal_cost_jacobian(self, x_term):
        jac = np.zeros((1, x_term.shape[0]))
        return jac

    def stage_ineq_constraint_func(self, xu_stage):
        return np.array([])

    def stage_ineq_constraint_jacobian(self, xu_stage):
        return np.array([])


class VirtualPowerPlant(Model):
    def __init__(self, system_cost_weight, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon,
                 n_simulation_steps, n_batteries, n_tcls, n_inverted_pendulums):

        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps)
        self.is_nonlinear = True
        self.is_discrete = True

        # DESIGN SYSTEM START #
        self.devices = [BatteryModel(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon,
                                     n_simulation_steps, idx=i) for i in range(n_batteries)] \
                       + [TCLModel(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon,
                                   n_simulation_steps, idx=i) for i in range(n_tcls)] \
                       + [InvertedPendulumModel(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon,
                                                n_simulation_steps, idx=i) for i in range(n_inverted_pendulums)]

        # DESIGN SYSTEM STOP #

        self.device_zstage_indices = []
        self.device_xterm_indices = []
        self.device_xstage_indices = []
        self.device_ustage_indices = []
        self.device_xustage_indices = []
        self.device_wstage_indices = []
        for d, device in enumerate(self.devices):
            # given x_term of all devices, which indices refer to device d
            idx_start = int(np.sum([self.devices[dd].n_states for dd in range(d)]))
            idx_stop = idx_start + device.n_states
            self.device_xterm_indices.append(list(range(idx_start, idx_stop)))

            # given x_stage of all devices, which indices refer to device d
            idx_start = int(np.sum([self.devices[dd].n_states * (self.state_lag + 1) for dd in range(d)]))
            idx_stop = idx_start + (device.n_states * (self.state_lag + 1))
            self.device_xstage_indices.append(list(range(idx_start, idx_stop)))

            # given u_stage of all devices, which indices refer to device d
            idx_start = int(np.sum([self.devices[dd].n_states * (self.state_lag + 1) for dd in range(len(self.devices))]
                                   + [self.devices[dd].n_inputs * (self.input_lag + 1) for dd in range(d)]))
            idx_stop = idx_start + (device.n_inputs * (self.input_lag + 1))
            self.device_ustage_indices.append(list(range(idx_start, idx_stop)))

            # given w_stage of all devices, which indices refer to device d
            idx_start = int(np.sum([self.devices[dd].n_states * (self.state_lag + 1)
                                    + self.devices[dd].n_inputs * (self.input_lag + 1) for dd in
                                    range(len(self.devices))]
                                   + [self.devices[dd].n_disturbances * (self.disturbance_lag + 1) for dd in range(d)]))
            idx_stop = idx_start + (device.n_disturbances * (self.disturbance_lag + 1))
            self.device_wstage_indices.append(list(range(idx_start, idx_stop)))

            # given z_lagged of all devices, which indices refer to zstage of device d
            self.device_xustage_indices.append(self.device_xstage_indices[-1] + self.device_ustage_indices[-1])
            self.device_zstage_indices.append(self.device_xstage_indices[-1] + self.device_ustage_indices[-1]
                                              + self.device_wstage_indices[-1])

        self.system_variables()
        self.set_indices()
        self.set_ref_state()
        self.set_init_state()
        self.parameters['R'] = system_cost_weight

        self.state_var_funcs = None
        self.next_state_funcs = None
        self.true_next_state_funcs = None
        self.next_state_jacobians = None

    def state_var_func(self, z_lagged, k=0, return_std=False):
        self.set_simulation_step(k)
        state_var = []

        for d in range(len(self.devices)):
            for func in self.state_var_funcs[d]:
                state_var.append(func(z_lagged[self.device_zstage_indices[d]], return_std=return_std))

        state_var = np.array(state_var)
        if return_std:
            return state_var[:, 0], state_var[:, 1]
        else:
            return state_var

    def next_state_prior_func(self, z_lagged, k=0):
        # first 0 index fetches the mean, second 0 index fetches the first row of the return predictions
        self.set_simulation_step(k)
        next_state_prior = []

        for d in range(len(self.devices)):
            next_state_prior.append(self.devices[d].next_state_prior_func(z_lagged[self.device_zstage_indices[d]]))

        next_state_prior = np.array(next_state_prior)

        return next_state_prior

    def next_state_func(self, z_lagged, k=0, return_std=False, true_system=False):
        # first 0 index fetches the mean, second 0 index fetches the first row of the return predictions
        self.set_simulation_step(k)
        next_state = []
        funcs = self.next_state_funcs if not true_system else self.true_next_state_funcs

        for d in range(len(self.devices)):
            for func in funcs[d]:
                next_state.append(func(z_lagged[self.device_zstage_indices[d]], return_std=return_std))

        next_state = np.array(next_state)
        if return_std:
            return next_state[:, 0], next_state[:, 1]
        else:
            return next_state

    def next_state_jacobian(self, z_stage, k=0):
        self.set_simulation_step(k)
        jac = np.zeros((self.n_states, z_stage.shape[0]))
        state_start_idx = 0
        for d in range(len(self.devices)):
            for state_jac in self.next_state_jacobians[d]:
                state_end_idx = state_start_idx + 1
                jac[state_start_idx:state_end_idx, self.device_zstage_indices[d]] \
                    = jac[state_start_idx:state_end_idx, self.device_zstage_indices[d]] \
                      + state_jac(z_stage[self.device_zstage_indices[d]])
                state_start_idx = state_end_idx

        return jac

    def stage_ineq_constraint_func(self, xu_stage, k, is_stage):
        self.set_simulation_step(k)
        cons = self.system_stage_ineq_constraint_func(xu_stage)

        for d, dev in enumerate(self.devices):
            cons = np.append(cons,
                             dev.stage_ineq_constraint_func(
                                 xu_stage[self.device_xustage_indices[d] if is_stage
                                 else self.device_xterm_indices[d]]))
        return cons

    def stage_ineq_constraint_jacobian(self, xu_stage, k, is_stage):
        self.set_simulation_step(k)
        jac = self.system_stage_ineq_constraint_jacobian(xu_stage)

        n_features = xu_stage.shape[0]
        for d, dev in enumerate(self.devices):
            dev_jac = dev.stage_ineq_constraint_jacobian(
                xu_stage[self.device_xustage_indices[d] if is_stage else self.device_xterm_indices[d]])

            new_jac = np.zeros((dev_jac.shape[0], n_features))

            new_jac[:, self.device_xustage_indices[d] if is_stage else self.device_xterm_indices[d]] \
                = new_jac[:, self.device_xustage_indices[d] if is_stage else self.device_xterm_indices[d]] + dev_jac
            jac = np.vstack([jac, new_jac])

        return jac

    def linarized_next_state_prior_func(self, z, z_bar, k=0):
        self.set_simulation_step(k)
        next_state = []
        for d, dev in enumerate(self.devices):
            next_state.append(dev.linearized_next_state_prior_func(z[self.device_zstage_indices[d]],
                                                                   z_bar[self.device_zstage_indices[d]]))

        return np.array(next_state)

    def true_next_state_func(self, z_lagged, k, is_synthetic_data=False, return_std=False):
        if is_synthetic_data:
            self.set_simulation_step(k)

            mean = np.concatenate([self.devices[d].true_next_state_func(z_lagged[self.device_zstage_indices[d]])
                                   for d in range(len(self.devices))])
            if return_std:
             return (mean, 0)
            else:
                return mean
        else:
            return self.next_state_func(z_lagged, k, return_std=return_std, true_system=True)

    def set_simulation_step(self, k):
        self.simulation_step = k
        for dev in self.devices:
            dev.set_simulation_step(k)

    def P_out_stage(self, z_stage):
        P_out = np.array([0])
        for d, device in enumerate(self.devices):
            P_out = P_out + device.P_out_stage(z_stage[self.device_zstage_indices[d]])

        return P_out

    def P_out_stage_jacobian(self, z_stage):

        jac = np.zeros((1, z_stage.shape[0]))
        for d, device in enumerate(self.devices):
            jac[0, self.device_zstage_indices[d]] \
                = device.P_out_stage_jacobian(z_stage[self.device_zstage_indices[d]])

        return jac

    def P_out_term(self, x_term):
        P_out = 0
        for d, device in enumerate(self.devices):
            P_out += device.P_out_term(x_term[self.device_xterm_indices[d]])

        return P_out

    def P_out_term_jacobian(self, x_term):
        jac = np.zeros((1, x_term.shape[0]))
        for d, device in enumerate(self.devices):
            jac[0, self.device_xterm_indices[d]] \
                = device.P_out_term_jacobian(x_term[self.device_xterm_indices[d]])

        return jac

    def system_stage_cost_func(self, z_stage):
        return self.parameters['R'] * (self.P_out_stage(z_stage) - self.parameters['P_ref'][self.simulation_step]) ** 2

    def system_stage_cost_jacobian(self, z_stage):
        # jac = np.zeros((1, z_stage.shape[0]))
        # return jac
        return 2 * self.parameters['R'] * (self.P_out_stage(z_stage) - self.parameters['P_ref'][self.simulation_step]) \
               * self.P_out_stage_jacobian(z_stage)

    def system_terminal_cost_func(self, x_term):
        return (self.P_out_term(x_term) - 0) ** 2
        # return 0

    def system_terminal_cost_jacobian(self, x_term):
        # jac = np.zeros((1, x_term.shape[0]))
        # return jac
        return 2 * (self.P_out_term(x_term) - 0) * self.P_out_term_jacobian(x_term)

    def true_stage_cost_func(self, z_stage):

        stage_cost = self.system_stage_cost_func(z_stage)
        for d, device in enumerate(self.devices):
            stage_cost = stage_cost + device.true_stage_cost_func(z_stage[self.device_zstage_indices[d]])

        return stage_cost

    def true_stage_cost_jacobian(self, z_stage):

        stage_cost_jac = self.system_stage_cost_jacobian(z_stage)
        for d, device in enumerate(self.devices):
            stage_cost_jac[:, self.device_zstage_indices[d]] += device.true_stage_cost_jacobian(
                z_stage[self.device_zstage_indices[d]])

        return stage_cost_jac

    def terminal_cost_func(self, x_term):

        terminal_cost = self.system_terminal_cost_func(x_term)
        for d, device in enumerate(self.devices):
            terminal_cost += device.terminal_cost_func(x_term[self.device_xterm_indices[d]])

        return terminal_cost

    def terminal_cost_jacobian(self, x_term):

        terminal_cost_jac = self.system_terminal_cost_jacobian(x_term)
        for d, device in enumerate(self.devices):
            terminal_cost_jac[:, self.device_xterm_indices[d]] += device.terminal_cost_jacobian(
                x_term[self.device_xterm_indices[d]])

        return terminal_cost_jac

    def system_variables(self):
        # define system (measurable) states over given lag
        self.states = np.concatenate([device.states for device in self.devices])
        # self.n_states = np.sum([device.n_states for device in self.devices])

        self.inputs = np.concatenate([device.inputs for d, device in enumerate(self.devices)])
        # self.n_inputs = np.sum([device.n_inputs for device in self.devices])

        self.disturbances = np.concatenate([device.disturbances for d, device in enumerate(self.devices)])
        # self.n_disturbances = np.sum([device.n_disturbances for device in self.devices])

        self.outputs = np.concatenate([device.outputs for d, device in enumerate(self.devices)])
        # self.n_outputs = np.sum([device.n_outputs for device in self.devices])

        # define system parameters
        self.parameters = {'P_ref': [-32 for i in range(int(self.n_simulation_steps / 4))]
                                     + [-24 for i in range(int(self.n_simulation_steps / 4))]
                                     + [-32 for i in range(int(self.n_simulation_steps / 4))]
                                     + [-24 for i in range(int(self.n_simulation_steps / 4))]
                                     + [-32 for i in range(self.n_horizon + self.n_simulation_steps
                                                           - 4 * int(self.n_simulation_steps / 4))],
                           # 'P_ref': -48 * np.ones(self.n_simulation_steps + self.n_horizon),
                           'R': 0.01}

        self.numerical_bounds = {}
        self.numerical_gaussian = {}
        for device in self.devices:
            self.parameters.update(device.parameters)
            self.numerical_bounds.update(device.numerical_bounds)
            # self.numerical_gaussian.update(device.numerical_gaussian)

    def stage_bounds(self, z_stage):
        # stage_bounds = np.vstack([device.stage_bounds(z_stage) for device in self.devices])
        stage_bounds = np.zeros((self.n_states + self.n_inputs, 2))
        for d, dev in enumerate(self.devices):
            stage_bounds[self.device_xstage_indices[d] + self.device_ustage_indices[d], :] \
                = dev.stage_bounds(z_stage[self.device_zstage_indices[d]])
        return stage_bounds

    def term_bounds(self, x_term):
        # term_bounds = np.vstack([device.term_bounds(x_term) for device in self.devices])
        term_bounds = np.zeros((self.n_states, 2))
        for d, dev in enumerate(self.devices):
            term_bounds[self.device_xterm_indices[d]] \
                = dev.term_bounds(x_term[self.device_xterm_indices[d]])
        return term_bounds

    def set_ref_state(self):
        self.ref_state = np.concatenate([device.ref_state for device in self.devices])

    def set_init_state(self):
        self.init_state = np.concatenate([device.init_state for device in self.devices])

    def system_stage_ineq_constraint_func(self, xu_stage):
        cons = []
        # for d in range(len(self.devices)):
        #     for dd in range(d + 1, len(self.devices)):
        #         # if this is a TCL, make sure that all 'distributed' inputs T_a are equivalent
        #         if 'T_a' in self.devices[d].inputs and 'T_a' in self.devices[dd].inputs:
        #             T_a_i = xu_stage[self.device_ustage_indices[d][1]]
        #             T_a_j = xu_stage[self.device_ustage_indices[dd][1]]
        #
        #             cons.append(T_a_i - T_a_j)
        #             cons.append(T_a_j - T_a_i)
        return np.array(cons)

    # def stage_ineq_constraint_func(self, xu_stage):
    #     cons = self.system_stage_ineq_constraint_func(xu_stage)
    #     for d, dev in enumerate(self.devices):
    #         cons.append(dev.stage_ineq_constraint_func(xu_stage[self.device_xstage_indices[d]]))
    #         # cons.append(dev.stage_ineq_constraint_func(xu_stage[self.device_xterm_indices[d]]))
    #
    #     return np.array(cons)

    # def stage_ineq_constraint_jacobian(self, xu_stage):
    #     jacobian = self.system_stage_cost_jacobian(xu_stage)
    #     for d, dev in enumerate(self.devices):
    #         jacobian[:, self.device_xterm_indices[d]] \
    #             = jacobian[:, self.device_xterm_indices[d]] \
    #               + dev.stage_ineq_constraint_jacobian(xu_stage[self.device_xstage_indices[d]])
    #
    #         # jacobian[:, self.device_xterm_indices[d]] \
    #         #     = jacobian[:, self.device_xterm_indices[d]] \
    #         #       + dev.stage_ineq_constraint_jacobian(xu_stage[self.device_xterm_indices[d]])
    #
    #     return jacobian

    def system_stage_ineq_constraint_jacobian(self, xu_stage):

        jacobian = np.zeros((0, len(xu_stage)))
        # for d in range(len(self.devices)):
        #     for dd in range(d + 1, len(self.devices)):
        #         # if this is a TCL, make sure that all 'distributed' inputs T_a are equivalent
        #         if 'T_a' in self.devices[d].inputs and 'T_a' in self.devices[dd].inputs:
        #             T_a_i_idx = self.device_ustage_indices[d][1]
        #             T_a_j_idx = self.device_ustage_indices[dd][1]
        #
        #             jacobian = np.vstack([jacobian, np.zeros((2, len(xu_stage)))])
        #
        #             jacobian[-2, T_a_i_idx] = 1
        #             jacobian[-2, T_a_j_idx] = -1
        #
        #             jacobian[-1, T_a_j_idx] = 1
        #             jacobian[-1, T_a_i_idx] = -1

        return jacobian


class InvertedPendulumModel(Model):

    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps, idx):
        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps)
        self.is_nonlinear = True
        self.is_discrete = False

        self.name = 'IP'
        self.idx = idx
        self.system_variables()
        self.set_indices()
        self.set_ref_state()
        self.set_init_state()

        self.g = self.parameters['g']
        self.l = self.parameters['l']
        self.eta = self.parameters['eta']
        self.m = self.parameters['m']

    def true_next_state_discrete_func(self, z_lagged):
        theta = z_lagged[self.x_stage_indices[0]]
        theta_dot = z_lagged[self.x_stage_indices[1]]
        u1 = z_lagged[self.u_stage_indices[0]]

        return np.array([theta_dot, theta ** 2 + u1])

    def theta_dot(self, z_lagged):
        theta_dot = z_lagged[self.x_stage_indices[1]]
        return theta_dot

    def true_theta_dot_dot(self, z_lagged):
        theta = z_lagged[self.x_stage_indices[0]]
        theta_dot = z_lagged[self.x_stage_indices[1]]
        u1 = z_lagged[self.u_stage_indices[0]]
        w1 = z_lagged[self.w_stage_indices[0]]
        sin_theta = np.sin(theta)
        # NOTE: must round result to avoid precision errors when computing jacobian_func, in which a zero difference
        # is not nulled

        res = ((self.g / self.l) * sin_theta) \
              - ((self.eta / (self.m * self.l ** 2)) * theta_dot) \
              + ((1 / (self.m * self.l ** 2)) * (u1 + w1))

        return res

    def true_next_state_cont_func(self, z_lagged):
        state_dot = np.hstack([self.theta_dot(z_lagged),
                               self.true_theta_dot_dot(z_lagged)])
        return state_dot

    def linear_discretized_func(self, z_lagged):
        p = self.prior_parameters

        # linearize entire function
        theta_ss = 0

        A = np.array([[0, 1], [(p['g'] / p['l']) * np.cos(theta_ss), -(p['eta'] / (p['m'] * p['l'] ** 2))]])
        B = np.array([[0, 0], [1 / (p['m'] * p['l'] ** 2), 1 / (p['m'] * p['l'] ** 2)]])

        A_d = np.eye(A.shape[0]) + (A * self.t_step)
        B_d = B * self.t_step

        next_state = A_d @ z_lagged[self.x_stage_indices] \
                     + B_d @ z_lagged[self.u_stage_indices + self.w_stage_indices]

        return next_state

    def terminal_cost_func(self, x_term):
        theta = x_term[self.x_stage_indices[0]]
        cost = theta ** 2
        return cost

    def terminal_cost_jacobian(self, x_term):
        theta_idx = self.x_stage_indices[0]
        theta = x_term[theta_idx]

        jac = np.zeros((1, x_term.shape[0]))
        jac[0, theta_idx] = 2 * theta

        return jac

    def true_stage_cost_func(self, z_stage, return_std=False):
        """
        :param x: n_y * (n_horizon - 1) array of stage outputs
        :param u: n_u * (n_horizon - 1) array of stage inputs
        :return:
        """
        theta = z_stage[self.x_stage_indices[0]]
        res = [theta ** 2]
        if return_std:
            return res, [0]
        else:
            return res

    def stage_bounds(self, z_stage):
        # stage_bounds = {f'theta_{self.idx}':
        #                     [self.parameters['theta_min'][self.simulation_step],
        #                      self.parameters['theta_max'][self.simulation_step]],
        #                 f'theta_dot_{self.idx}':
        #                     [self.parameters['theta_dot_min'][self.simulation_step],
        #                      self.parameters['theta_dot_max'][self.simulation_step]],
        #                 'u': [self.parameters['u_min'][self.simulation_step],
        #                        self.parameters['u_max'][self.simulation_step]]}

        stage_bounds = {f'theta_{self.idx}': [-np.pi, np.pi],
                        f'theta_dot_{self.idx}': [-np.pi, np.pi],
                        'u': [-1, 1]}
        return np.vstack(list(stage_bounds.values()))

    def term_bounds(self, x_term):
        # term_bounds = {f'theta_{self.idx}':
        #                    [self.parameters['theta_min'][self.simulation_step],
        #                     self.parameters['theta_max'][self.simulation_step]],
        #                f'theta_dot_{self.idx}':
        #                    [self.parameters['theta_min'][self.simulation_step],
        #                     self.parameters['theta_max'][self.simulation_step]]}

        term_bounds = {f'theta_{self.idx}': [-np.pi, np.pi],
                       f'theta_dot_{self.idx}': [-np.pi, np.pi]}
        return np.vstack(list(term_bounds.values()))

    def system_variables(self):
        # define system (measurable) states over given lag

        self.states = [f'theta_{self.idx}', f'theta_dot_{self.idx}']

        self.inputs = [f'u_{self.idx}']

        # define system measurements
        # measurements = states.copy()

        self.disturbances = [f'w_{self.idx}']
        self.outputs = []
        # define system parameters

        self.original_parameters = {'m': 0.15, 'l': 0.5, 'g': 9.81, 'eta': 0.1}
                                    # 'theta_min': -np.pi * np.ones(self.n_simulation_steps + self.n_horizon),
                                    # 'theta_max': np.pi * np.ones(self.n_simulation_steps + self.n_horizon),
                                    # 'theta_dot_min': -np.inf * np.ones(self.n_simulation_steps + self.n_horizon),
                                    # 'theta_dot_max': np.inf * np.ones(self.n_simulation_steps + self.n_horizon),
                                    # 'u_min': -1 * np.ones(self.n_simulation_steps + self.n_horizon),
                                    # 'u_max': 1 * np.ones(self.n_simulation_steps + self.n_horizon)
                                    # }  # {'m': 0.15, 'l': 0.5, 'g': 9.81, 'eta': 0.1}


        self.parameters = dict(self.original_parameters)
        self.prior_parameters = self.parameters

        self.numerical_bounds = {f'theta_{self.idx}': [-2 * np.pi, 2 * np.pi], # todo
                                 f'theta_dot_{self.idx}': [-2 * np.pi, 2 * np.pi],
                                 f'u_{self.idx}': [-1, 1],
                                 f'w_{self.idx}': [-1, 1]}

        self.numerical_gaussian = {f'theta_{self.idx}': {'mean': 0, 'std': np.pi},
                                   f'theta_dot_{self.idx}': {'mean': 0, 'std': 10 * np.pi},
                                   f'u_{self.idx}': {'mean': 0, 'std': 1}}

        self.disturbance_train = np.zeros((self.n_horizon + self.n_simulation_steps, 1))
        self.disturbance_train[int(self.n_simulation_steps * 0.25):int(self.n_simulation_steps * 0.25) + 3] = -0.5
        self.disturbance_train[int(self.n_simulation_steps * 0.5):int(self.n_simulation_steps * 0.5) + 3] = 0.5
        self.disturbance_train[int(self.n_simulation_steps * 0.75):int(self.n_simulation_steps * 0.75) + 3] = -0.5

    def linear_theta_dot(self, z_lagged):
        theta_dot = z_lagged[self.x_stage_indices[1]]
        u1 = z_lagged[self.u_stage_indices[0]]
        return -((self.eta / (self.m * self.l ** 2)) * theta_dot) + ((1 / (self.m * self.l ** 2)) * u1)

    # def next_state_prior_func(self, z_lagged):
    #     # z_lagged = [x_lagged, u_lagged, w_lagged]
    #     # reduce mass and neglect friction
    #     # self.prior_parameters = {'m': 0.1, 'l': 0.5, 'g': 9.81, 'eta': 0.0}
    #     # res = self.linear_discretized_func(z)
    #     res = self.rk_discretized_func(z_lagged, cont_func=self.linear_theta_dot)
    #     return res
    #
    # def next_state_prior_jacobian(self, z_lagged):
    #     return jacobian_func(self.next_state_prior_func, z_lagged)

    def next_state_prior_func(self, z_lagged):
        theta = z_lagged[self.x_stage_indices[0]]
        theta_dot = z_lagged[self.x_stage_indices[1]]
        return np.array([theta, theta_dot])
        # return np.array([0])

    def linearized_next_state_prior_func(self, z_lagged, z_lagged_bar):
        theta = z_lagged[self.x_stage_indices[0]]
        theta_dot = z_lagged[self.x_stage_indices[1]]
        return np.array([theta, theta_dot])

    def linearized_next_state_prior_jacobian(self, z_lagged, z_lagged_bar):
        return np.array([[1, 0], [0, 1]])

    def next_state_prior_jacobian(self, z_lagged):
        jac = np.zeros((self.n_states, z_lagged.shape[0]))
        theta_idx = self.x_stage_indices[0]
        theta_dot_idx = self.x_stage_indices[1]
        jac[0, theta_idx] = 1
        jac[1, theta_dot_idx] = 1
        return jac

    def set_ref_state(self):
        self.ref_state = np.array([0, 0])

    def set_init_state(self):
        self.init_state = np.array([1, 0])

    def stage_ineq_constraint_func(self, xu_stage):
        # theta = xu_stage[self.x_stage_indices[0]]
        # g = [theta - (np.pi),
             # -theta - (np.pi)]
        g = []
        return np.array([g])

    def stage_ineq_constraint_jacobian(self, xu_stage):
        # theta_idx = self.x_stage_indices[0]
        jac = np.zeros((0, xu_stage.shape[0]))
        # jac[0, theta_idx] = 1
        # jac[1, theta_idx] = -1
        return jac

    def stage_cost_prior_func(self, z_stage):
        return np.array([0])

    def stage_cost_prior_jacobian(self, z_stage):
        jac = np.zeros((1, z_stage.shape[0]))
        return jac

    def true_stage_cost_jacobian(self, z_stage):
        theta = z_stage[self.x_stage_indices[0]]
        grad = [2 * theta, 0, 0, 0]
        return np.vstack([grad])

    def terminal_cost_jacobian(self, x_term):
        theta = x_term[0]
        grad = [2 * theta, 0]
        return np.vstack([grad])

    def P_out_stage(self, z_stage):
        return 0

    def P_out_stage_jacobian(self, z_stage):
        jac = np.zeros((1, z_stage.shape[0]))
        return jac

    def P_out_term(self, x_term):
        return 0

    def P_out_term_jacobian(self, x_term):
        jac = np.zeros((1, x_term.shape[0]))
        return jac
