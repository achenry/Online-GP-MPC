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
        self.disturbances = None
        self.outputs = None
        self.parameters = None

        self.x_stage_indices = None
        self.u_stage_indices = None
        self.w_stage_indices = None
        self.y_stage_indices = None

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
        self.simulation_step = None

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

    def discrete_func(self, z_lagged):
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
            cont_func = self.cont_func

        k1 = cont_func(z_lagged)

        z_delta = np.array(z_lagged)
        z_delta[self.x_stage_indices] = z_delta[self.x_stage_indices] + (self.t_step * (k1 / 2))
        k2 = cont_func(z_delta)

        z_delta = np.array(z_lagged)
        z_delta[self.x_stage_indices] = z_delta[self.x_stage_indices] + (self.t_step * (k2 / 2))
        k3 = cont_func(z_delta)

        z_delta = np.array(z_lagged)
        z_delta[self.x_stage_indices] = z_delta[self.x_stage_indices] + (self.t_step * k3)
        k4 = cont_func(z_lagged)

        next_state = z_lagged[self.x_stage_indices] + ((1 / 6) * (k1 + (2 * k2) + (2 * k3) + k4))

        return next_state

    def true_next_state_func(self, z_lagged):

        if self.is_discrete:
            return self.discrete_func(z_lagged)
        elif self.is_nonlinear:
            return self.rk_discretized_func(z_lagged)
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

    def stage_constraint_func(self, z_stage):
        stage_bounds = self.stage_bounds(z_stage)
        state_bounds = stage_bounds[:self.n_states]
        input_bounds = stage_bounds[self.n_states:]

        bound_constraints = self.set_bound_constraints(state_bounds=state_bounds, input_bounds=input_bounds,
                                                       x=z_stage[self.x_stage_indices],
                                                       u=z_stage[self.u_stage_indices])

        nonbound_constraints = np.array([])

        return np.concatenate([bound_constraints, nonbound_constraints])

    def term_constraint_func(self, x_term):
        state_bounds = self.term_bounds(x_term)
        bound_constraints = self.set_bound_constraints(state_bounds=state_bounds, x=x_term)

        nonbound_constraints = np.array([])

        return np.concatenate([bound_constraints, nonbound_constraints])

    def set_simulation_step(self, k):
        self.simulation_step = k


class TCLModel(Model):
    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps, idx=1):
        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps)

        self.is_nonlinear = True
        self.is_discrete = True

        self.idx = idx

        self.system_variables()
        self.set_indices()

        self.set_init_state()

    def system_variables(self):
        # define system (measurable) states over given lag

        self.states = [x + '_' + str(self.idx) for x in ['T_zone']]

        self.inputs = [f'm_dot_a_{self.idx}', 'T_a']

        self.disturbances = ['T_o', f'Q_s_{self.idx}', f'Q_i_{self.idx}']

        self.outputs = [y + '_' + str(self.idx) for y in ['P_out']]

        # define system parameters

        self.parameters = {'T_zone_ref': 25 * np.ones(self.n_simulation_steps)}

        self.numerical_bounds = {f'T_zone_{self.idx}': [21, 29],
                                 f'm_dot_a_{self.idx}': [0.2, 2.2],
                                 'T_a': [10, 16],
                                 'T_o': [20.6, 36.1],
                                 f'Q_s_{self.idx}': [0, 6.9],
                                 f'Q_i_{self.idx}': [0.17, 5.04]}

        self.numerical_gaussian = {f'T_zone_{self.idx}': {'mean': 24.2, 'std': 1.05},
                                 f'm_dot_a_{self.idx}': {'mean': 0.29, 'std': 0.13},
                                 'T_a': {'mean': 13, 'std': 0.9},
                                 'T_o': {'mean': 28.8, 'std': 3.7},
                                 f'Q_s_{self.idx}': {'mean': 1.5, 'std': 2},
                                 f'Q_i_{self.idx}': {'mean': 0.79, 'std': 0.71}}

    def set_init_state(self):
        self.init_state = np.array([20])

    def stage_bounds(self, z_stage):
        stage_bounds = {f'T_zone_{self.idx}': [21, 29],
                        f'm_dot_a_{self.idx}': [0.2, 2.2],
                        'T_a': [10, 16]}
        return np.vstack(list(stage_bounds.values()))

    def term_bounds(self, x_term):
        term_bounds = {f'T_zone_{self.idx}': [21, 29]}
        return np.vstack(list(term_bounds.values()))

    def next_state_prior_func(self, z_lagged):
        T_zone = 0# z_lagged[self.x_stage_indices[0]]
        return T_zone

    def next_state_prior_jacobian(self, z_lagged):
        jac = np.zeros((self.n_states, z_lagged.shape[0]))
        # T_zone_idx = self.x_stage_indices[0]
        # jac[0, T_zone_idx] = 1
        return jac

    def stage_cost_prior_func(self, z_stage):
        return 0

    def stage_cost_prior_jacobian(self, z_stage):
        jac = np.zeros((1, z_stage.shape[0]))
        return jac

    def P_out_stage(self, z_stage):
        m_dot_a = z_stage[self.u_stage_indices[0]]
        T_o = z_stage[self.w_stage_indices[0]]
        T_a = z_stage[self.u_stage_indices[1]]
        P_fan = 0.0076 * m_dot_a ** 3 + 4.8865
        P_chiller = z_stage[self.u_stage_indices[0]] * (T_o - T_a) if T_o > T_a else 0
        return - (P_fan + P_chiller)

    def P_out_stage_jacobian(self, z_stage):
        m_dot_a_idx = self.u_stage_indices[0]
        m_dot_a = z_stage[m_dot_a_idx]
        T_o_idx = self.w_stage_indices[0]
        T_o = z_stage[T_o_idx]
        T_a_idx = self.u_stage_indices[1]
        T_a = z_stage[T_a_idx]

        jac = np.zeros((1, z_stage.shape[0]))

        jac[0,  m_dot_a_idx] = 0.0076 * 3 * m_dot_a**2 + ((T_o - T_a) if T_o > T_a else 0)
        jac[0,  T_a_idx] = -m_dot_a if T_o > T_a else 0
        jac[0,  T_o_idx] = m_dot_a if T_o > T_a else 0
        return -jac

    def P_out_term(self, x_term):
        return 0

    def P_out_term_jacobian(self, x_term):
        jac = np.zeros((1, x_term.shape[0]))
        return jac

    def true_stage_cost_func(self, z_stage):
        T_zone = z_stage[self.x_stage_indices[0]]
        return (T_zone - self.parameters['T_zone_ref'][self.simulation_step]) ** 2

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
        return np.array([])

    def stage_ineq_constraint_jacobian(self, xu_stage):
        return np.array([])


class BatteryModel(Model):
    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps, idx=1):

        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps)
        self.is_nonlinear = True
        self.is_discrete = True

        self.idx = idx
        self.system_variables()
        self.set_indices()
        self.set_init_state()

    def system_variables(self):
        # define system (measurable) states over given lag

        self.states = [f'soc_{self.idx}']

        self.inputs = [f'P_{self.idx}']

        self.disturbances = [f'P_ref_{self.idx}']

        self.outputs = []

        # define system parameters

        self.parameters = {'P_ref': [5000 for i in range(int(self.n_simulation_steps / 4))]
                                    + [-5000 for i in range(int(self.n_simulation_steps / 4))]
                                    + [5000 for i in range(int(self.n_simulation_steps / 4))]
                                    + [-5000 for i in range(int(self.n_simulation_steps / 4))]
                                    + [0 for i in range(self.n_horizon + self.n_simulation_steps
                                                        - 4 * int(self.n_simulation_steps / 4))],
                           'P_min': -8593,
                           'P_max': 9170}

        self.numerical_bounds = {f'soc_{self.idx}': [0, 1],
                                 f'P_{self.idx}': [-8593, 9170],
                                 f'P_ref_{self.idx}': [-8593, 9170]}

        self.numerical_gaussian = {f'soc_{self.idx}': {'mean': 0.55, 'std': 0.011},
                                   f'P_{self.idx}': {'mean': -61, 'std': 918},
                                   f'P_ref_{self.idx}': {'mean': -61, 'std': 918}}

    def set_init_state(self):
        self.init_state = np.array([0.5])

    def stage_bounds(self, z_stage):

        P = z_stage[self.u_stage_indices[0]]
        stage_bounds = {'soc': [0, 1],
                        'P': [0, self.parameters['P_max']] if P >= 0 else [self.parameters['P_min'], 0]}

        return np.vstack(list(stage_bounds.values()))

    def term_bounds(self, x_term):
        term_bounds = {'soc': [0, 1]}
        return np.vstack(list(term_bounds.values()))

    def next_state_prior_func(self, z_lagged):
        # TODO
        soc_prior = z_lagged[self.x_stage_indices[0]]
        #soc_prior = 0
        return np.array([soc_prior])

    def next_state_prior_jacobian(self, z_lagged):
        jac = np.zeros((self.n_states, z_lagged.shape[0]))
        soc_idx = self.x_stage_indices[0]
        jac[0, soc_idx] = 1
        return jac

    def stage_cost_prior_func(self, z_stage):
        return 0

    def stage_cost_prior_jacobian(self, z_stage):
        jac = np.zeros((1, z_stage.shape[0]))
        return jac

    def P_out_stage(self, z_stage):
        return z_stage[self.u_stage_indices[0]]

    def P_out_stage_jacobian(self, z_stage):
        P_idx = self.u_stage_indices[0]
        jac = np.zeros((1, z_stage.shape[0]))
        jac[0,  P_idx] = 1
        return jac

    def P_out_term(self, x_term):
        return 0

    def P_out_term_jacobian(self, x_term):
        jac = np.zeros((1, x_term.shape[0]))
        return jac

    def true_stage_cost_func(self, z_stage):
        P = z_stage[self.u_stage_indices[0]]
        return (P - self.parameters['P_ref'][self.simulation_step]) ** 2

    def true_stage_cost_jacobian(self, z_stage):
        P_idx = self.u_stage_indices[0]
        P = z_stage[P_idx]
        jac = np.zeros((1, z_stage.shape[0]))
        jac[0,  P_idx] = 2 * (P - self.parameters['P_ref'][self.simulation_step])
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
    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps,
                 n_batteries, n_tcls, n_inverted_pendulums):
        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps)
        self.is_nonlinear = True
        self.is_discrete = True

        # DESIGN SYSTEM START #
        self.devices = [BatteryModel(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon,
                                     n_simulation_steps, idx=i+1) for i in range(n_batteries)] \
                       + [TCLModel(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon,
                                   n_simulation_steps, idx=i+1) for i in range(n_tcls)] \
                       + [InvertedPendulumModel(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon,
                                   n_simulation_steps, idx=i+1) for i in range(n_inverted_pendulums)]
        # DESIGN SYSTEM STOP #

        self.device_xterm_indices = []

        self.device_zstage_indices = []
        self.device_xstage_indices = []
        self.device_ustage_indices = []
        self.device_wstage_indices = []

        for d, device in enumerate(self.devices):
            idx_start = int(np.sum([self.devices[dd].n_states for dd in range(d)]))
            idx_stop = idx_start + device.n_states
            self.device_xterm_indices.append(list(range(idx_start, idx_stop)))

            idx_start = int(np.sum([self.devices[dd].n_states * (self.state_lag + 1) for dd in range(d)]))
            idx_stop = idx_start + (device.n_states * (self.state_lag + 1))

            self.device_xstage_indices.append(list(range(idx_start, idx_stop)))

            idx_start = int(np.sum([self.devices[dd].n_states * (self.state_lag + 1)
                                + self.devices[dd].n_inputs * (self.input_lag + 1) for dd in range(d)])
                            + device.n_states * (self.state_lag + 1))

            idx_stop = idx_start + (device.n_inputs * (self.input_lag + 1))

            self.device_ustage_indices.append(list(range(idx_start, idx_stop)))

            idx_start = int(np.sum([self.devices[dd].n_states * (self.state_lag + 1)
                                + self.devices[dd].n_inputs * (self.input_lag + 1)
                                + self.devices[dd].n_disturbances * (self.disturbance_lag + 1) for dd in range(d)])
                            + device.n_states * (self.state_lag + 1) + device.n_inputs * (self.input_lag + 1))

            idx_stop = idx_start + (device.n_disturbances * (self.disturbance_lag + 1))

            self.device_wstage_indices.append(list(range(idx_start, idx_stop)))

            self.device_zstage_indices.append(self.device_xstage_indices[-1] + self.device_ustage_indices[-1] +
                                                self.device_wstage_indices[-1])

        self.system_variables()
        self.set_indices()
        self.set_init_state()

    def set_simulation_step(self, k):
        self.simulation_step = k
        for dev in self.devices:
            dev.set_simulation_step(k)

    def P_out_stage(self, z_stage):
        P_out = 0
        for d, device in enumerate(self.devices):
            P_out += device.P_out_stage(
                z_stage[self.device_zstage_indices[d]])

        return P_out

    def P_out_stage_jacobian(self, z_stage):
        
        jac = np.zeros((1, z_stage.shape[0]))
        for d, device in enumerate(self.devices):
            jac[0,  self.device_zstage_indices[d]] \
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
            jac[0,  self.device_xterm_indices[d]] \
                = device.P_out_term_jacobian(x_term[self.device_xterm_indices[d]])

        return jac

    def system_stage_cost_func(self, z_stage):
        # z_stage = [x_stage, u_stage]
        # return (self.P_out_stage(z_stage) - self.parameters['P_ref'][self.simulation_step]) ** 2
        return 0

    def system_stage_cost_jacobian(self, z_stage):
        jac = np.zeros((1, z_stage.shape[0]))
        return jac
        # return 2 * (self.P_out_stage(z_stage) - self.parameters['P_ref'][self.simulation_step]) \
        #        * self.P_out_stage_jacobian(z_stage)


    def system_terminal_cost_func(self, x_term):
        # return (self.P_out_term(x_term) - self.parameters['P_ref'][-1]) ** 2
        return 0

    def system_terminal_cost_jacobian(self, x_term):
        jac = np.zeros((1, x_term.shape[0]))
        return jac
        # return 2 * (self.P_out_term(x_term) - self.parameters['P_ref'][self.simulation_step]) \
        #        * self.P_out_term_jacobian(x_term)

    def true_stage_cost_func(self, z_stage):

        stage_cost = self.system_stage_cost_func(z_stage)
        for d, device in enumerate(self.devices):
            print(device.simulation_step)
            stage_cost += device.true_stage_cost_func(z_stage[self.device_zstage_indices[d]])

        return stage_cost

    def true_stage_cost_jacobian(self, z_stage):

        stage_cost_jac = self.system_stage_cost_jacobian(z_stage)
        for d, device in enumerate(self.devices):
            stage_cost_jac += device.true_stage_cost_jacobian(z_stage[self.device_zstage_indices[d]])

        return stage_cost_jac

    def terminal_cost_func(self, x_term):

        terminal_cost = self.system_terminal_cost_func(x_term)
        for d, device in enumerate(self.devices):

            terminal_cost += device.terminal_cost_func(x_term[self.device_xterm_indices[d]])

        return terminal_cost

    def terminal_cost_jacobian(self, x_term):

        terminal_cost_jac = self.system_terminal_cost_jacobian(x_term)
        for d, device in enumerate(self.devices):
            terminal_cost_jac += device.terminal_cost_jacobian(x_term[self.device_xterm_indices[d]])

        return terminal_cost_jac

    def system_variables(self):
        # define system (measurable) states over given lag
        self.states = np.array([device.states for d, device in enumerate(self.devices)]).flatten()
        self.n_states = np.sum([device.n_states for device in self.devices])

        self.inputs = np.array([device.inputs for d, device in enumerate(self.devices)]).flatten()
        self.n_inputs = np.sum([device.n_inputs for device in self.devices])

        self.disturbances = np.array([device.disturbances for d, device in enumerate(self.devices)]).flatten()
        self.n_disturbances = np.sum([device.n_disturbances for device in self.devices])

        self.outputs = np.array([device.outputs for d, device in enumerate(self.devices)]).flatten()
        self.n_outputs = np.sum([device.n_outputs for device in self.devices])

        # define system parameters
        self.parameters = {'P_ref': 30 * np.ones(self.n_simulation_steps)}
        self.numerical_bounds = {}
        self.numerical_gaussian = {}
        for device in self.devices:
            self.parameters.update(device.parameters)
            self.numerical_bounds.update(device.numerical_bounds)
            self.numerical_gaussian.update(device.numerical_gaussian)

    def stage_bounds(self, z_stage):
        stage_bounds = np.vstack([device.stage_bounds(z_stage) for device in self.devices])
        return stage_bounds

    def term_bounds(self, x_term):
        term_bounds = np.vstack([device.term_bounds(x_term) for device in self.devices])
        return term_bounds

    def set_init_state(self):
        self.init_state = np.concatenate([device.init_state for device in self.devices])

    def stage_ineq_constraint_func(self, xu_stage):
        cons = []
        for d in range(len(self.devices)):
            for dd in range(d + 1, len(self.devices)):
                # if this is a TCL, make sure that all 'distributed' inputs T_a are equivalent
                if 'T_a' in self.devices[d].inputs and 'T_a' in self.devices[dd].inputs:
                    T_a_i = xu_stage[self.device_ustage_indices[d][1]]
                    T_a_j = xu_stage[self.device_ustage_indices[dd][1]]

                    cons.append(T_a_i - T_a_j)
                    cons.append(T_a_j - T_a_i)
        return np.array(cons)

    def stage_ineq_constraint_jacobian(self, xu_stage):
        
        jacobian = np.zeros((0, len(xu_stage)))
        for d in range(len(self.devices)):
            for dd in range(d + 1, len(self.devices)):
                # if this is a TCL, make sure that all 'distributed' inputs T_a are equivalent
                if 'T_a' in self.devices[d].inputs and 'T_a' in self.devices[dd].inputs:
                    T_a_i_idx = self.device_ustage_indices[d][1]
                    T_a_j_idx = self.device_ustage_indices[dd][1]
                    
                    jacobian = np.vstack([jacobian, np.zeros((2, len(xu_stage)))])

                    jacobian[-2, T_a_i_idx] = 1
                    jacobian[-2, T_a_j_idx] = -1

                    jacobian[-1, T_a_j_idx] = 1
                    jacobian[-1, T_a_i_idx] = -1

        return jacobian


class InvertedPendulumModel(Model):

    def __init__(self, t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps, idx):

        super().__init__(t_step, state_lag, input_lag, disturbance_lag, output_lag, n_horizon, n_simulation_steps)
        self.is_nonlinear = True
        self.is_discrete = False

        self.idx = idx
        self.system_variables()
        self.set_indices()
        self.set_init_state()

        self.g = self.parameters['g']
        self.l = self.parameters['l']
        self.eta = self.parameters['eta']
        self.m = self.parameters['m']

    def theta_dot(self, z_lagged):
        theta_dot = z_lagged[self.x_stage_indices[1]]
        return theta_dot

    def true_theta_dot_dot(self, z_lagged):
        theta = z_lagged[self.x_stage_indices[0]]
        theta_dot = z_lagged[self.x_stage_indices[1]]
        u1 = z_lagged[self.u_stage_indices[0]]
        sin_theta = np.sin(theta)
        # NOTE: must round result to avoid precision errors when computing jacobian_func, in which a zero difference
        # is not nulled

        res = ((self.g / self.l) * sin_theta) \
              - ((self.eta / (self.m * self.l ** 2)) * theta_dot) \
              + ((1 / (self.m * self.l ** 2)) * u1)

        return res

    def cont_func(self, z_lagged):
        state_dot = np.hstack([self.theta_dot(z_lagged),
                               self.true_theta_dot_dot(z_lagged)])
        return state_dot

    def linear_discretized_func(self, z_lagged):
        p = self.prior_parameters

        # linearize entire function
        theta_ss = 0

        # J = jacobian_func(cont_func, z)
        A = np.array([[0, 1], [(p['g'] / p['l']) * np.cos(theta_ss), -(p['eta'] / (p['m'] * p['l'] ** 2))]])
        B = np.array([[0], [1 / (p['m'] * p['l'] ** 2)]])

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
        jac[0,  theta_idx] = 2 * theta

        return jac

    def true_stage_cost_func(self, z_stage):
        """
        :param x: n_y * (n_horizon - 1) array of stage outputs
        :param u: n_u * (n_horizon - 1) array of stage inputs
        :return:
        """
        theta = z_stage[self.x_stage_indices[0]]
        cost = theta ** 2
        return cost

    def stage_bounds(self, z_stage):
        stage_bounds = {'theta': [-np.pi, np.pi],
                        'theta_dot': [-np.inf, np.inf],
                        'u1': [-1, 1]}
        return np.vstack(list(stage_bounds.values()))

    def term_bounds(self, x_term):
        term_bounds = {'theta': [-np.pi, np.pi],
                       'theta_dot': [-np.inf, np.inf]}
        return np.vstack(list(term_bounds.values()))

    def system_variables(self):
        # define system (measurable) states over given lag

        self.states = [f'theta_{self.idx}', f'theta_dot_{self.idx}']

        self.inputs = [f'u_{self.idx}']

        # define system measurements
        # measurements = states.copy()

        self.disturbances = []
        self.outputs = []
        # define system parameters

        self.parameters = {'m': 0.15, 'l': 0.5, 'g': 9.81, 'eta': 0.1}  # {'m': 0.15, 'l': 0.5, 'g': 9.81, 'eta': 0.1}

        self.numerical_bounds = {f'theta_{self.idx}': [-np.pi, np.pi],
                                 f'theta_dot_{self.idx}': [-3 * np.pi, 3 * np.pi],
                                 f'u_{self.idx}': [-1, 1]}

        self.numerical_gaussian = {f'theta_{self.idx}': {'mean': 0, 'std': np.pi},
                                   f'theta_dot_{self.idx}': {'mean': 0, 'std': 10 * np.pi},
                                    f'u_{self.idx}': {'mean': 0, 'std': 1}}

    def linear_theta_dot(self, z_lagged):
        theta_dot = z_lagged[self.x_stage_indices[1]]
        u1 = z_lagged[self.u_stage_indices[0]]
        return -((self.eta / (self.m * self.l ** 2)) * theta_dot) + ((1 / (self.m * self.l ** 2)) * u1)

    def next_state_prior_func(self, z_lagged):
        # z_lagged = [x_lagged, u_lagged, w_lagged]
        # reduce mass and neglect friction
        # self.prior_parameters = {'m': 0.1, 'l': 0.5, 'g': 9.81, 'eta': 0.0}
        # res = self.linear_discretized_func(z)
        res = self.rk_discretized_func(z_lagged, cont_func=self.linear_theta_dot)

        return res

    def next_state_prior_jacobian(self, z_lagged):
        return jacobian_func(self.next_state_prior_func, z_lagged)

    def set_init_state(self):
        self.init_state = np.array([1, 1])

    def stage_ineq_constraint_func(self, xu_stage):
        return np.array([])

    def stage_ineq_constraint_jacobian(self, xu_stage):
        # return vertically stacked gradient_func functions for g_1(x_1), g_2(x_2), ..., g_N(x_N)
        return np.zeros((0, xu_stage.shape[0]))

    def stage_cost_prior_func(self, z_stage):
        return 0

    def stage_cost_prior_jacobian(self, z_stage):
        jac = np.zeros((1, z_stage.shape[0]))
        return jac

    def true_stage_cost_jacobian(self, z_stage):
        theta = z_stage[self.x_stage_indices[0]]
        grad = [2 * theta, 0]
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
