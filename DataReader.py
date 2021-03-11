import numpy as np
import pandas as pd


class DataReader:
    """
    take noisy measurements from actual functions or take infeed of measurements from external sources
    output feed of training data
    """

    def __init__(self, input_params, n_states, n_inputs, n_disturbances, n_outputs, n_simulation_steps):
        self.n_test_samples = input_params['n_test_samples']

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_disturbances = n_disturbances
        self.n_outputs = n_outputs

        self.state_lag = input_params['state_lag']
        self.input_lag = input_params['input_lag']
        self.disturbance_lag = input_params['disturbance_lag']
        self.output_lag = input_params['output_lag']
        self.n_horizon = input_params['n_horizon']

        self.unknown_system_model = None
        self.known_stage_cost = None
        self.known_terminal_cost = None
        self.unknown_stage_cost = None
        self.unknown_terminal_cost = None
        self.n_simulation_steps = n_simulation_steps

    def read_true_data(self, n_horizon, unknown_next_state_funcs, mpc_t_step):

        paths = []
        state_cols = []
        disturbance_cols = []
        sampling_t_steps = []
        for device_funcs in unknown_next_state_funcs:
            for func in device_funcs:
                if func['use_gp'] and func['training_data_path'] is not None:
                    paths.append(func['training_data_path'])
                    state_cols.append(func['state_cols'])
                    disturbance_cols.append(func['disturbance_cols'])
                    sampling_t_steps.append(func['sampling_t_step'])

        states = np.zeros((self.n_simulation_steps + n_horizon, 0))
        disturbances = np.zeros((self.n_simulation_steps + n_horizon, 0))

        for path, x_cols, w_cols, sampling_t_step in zip(paths, state_cols, disturbance_cols, sampling_t_steps):
            df = pd.read_csv(path, usecols=x_cols + w_cols)
            n_rows = len(df.index)
            indices = []
            while len(indices) < self.n_simulation_steps + n_horizon:
                n_samples_required = int((self.n_simulation_steps + n_horizon + 1 - len(indices)) # need this many more simulation indices
                                                 * mpc_t_step / sampling_t_step)

                # if n_samples_required == 0:
                #     new_sampling_indices = range(1)
                # else:
                new_sampling_indices = range(min(n_samples_required, # so need less sampling indices (for mpc_t_step finer than sampling t_step)
                                        int(n_rows * sampling_t_step / mpc_t_step))) # but only this many siulation indices are available in the sampling df

                new_simulation_indices = [[i for rep in range(int(sampling_t_step / mpc_t_step))]
                                          for i in new_sampling_indices]
                new_simulation_indices = list(np.concatenate(new_simulation_indices)) if len(new_simulation_indices) \
                    else []
                if not len(new_simulation_indices):
                    break
                indices = indices + new_simulation_indices

            indices = indices[:self.n_simulation_steps + n_horizon]

            states = np.hstack([states, df[x_cols].iloc[indices].values]) if len(indices) else []
            disturbances = np.hstack([disturbances, df[w_cols].iloc[indices].values]) if len(indices) else []

        return states, disturbances

    def read_training_data(self, func, run_gp, mpc_t_step, test_offset=None, zero_prior=False):

        path = func['training_data_path']
        input_cols = func['input_cols']
        state_cols = func['state_cols']
        disturbance_cols = func['disturbance_cols']
        output_state_col = func['output_state_col']
        sampling_t_step = func['sampling_t_step']
        # t_step = func['sampling_t_step']

        # function to read training data from csv files
        true_data = pd.read_csv(path, usecols=state_cols + input_cols + disturbance_cols + output_state_col) # + ['isgood'])

        # indices = np.random.randint(0, true_data.index[-1], func['max_n_training_samples'])

        n_training_samples = func['max_n_training_samples'] if run_gp else \
            np.max([func['max_n_training_samples'], func['n_init_training_samples']
                    + int(self.n_simulation_steps * (mpc_t_step / sampling_t_step))])
        if test_offset is None:
            n_test_samples = np.min([n_training_samples, self.n_test_samples])

            n_training_samples_set = int(n_training_samples / n_test_samples)
            # training_indices = np.concatenate([k + np.arange(k * n_training_samples_set, (k + 1) * n_training_samples_set)
            #                     for k in range(int((func['max_n_training_samples']) /
            #                                        (n_training_samples_set + 1)))])
                                               # int((func['max_n_training_samples']
                                               #    - (n_training_samples_set * self.n_test_samples))
                                               # / n_training_samples_set))])

            training_indices = np.arange(n_training_samples)# + n_test_samples)
            test_indices = np.arange(n_training_samples_set, n_training_samples - 2, # + n_test_samples - 1,
                                     n_training_samples_set + 1)
            self.n_test_samples = len(test_indices)
            training_indices = np.delete(training_indices, test_indices)
            # test_indices = np.concatenate([(k + 1) * n_training_samples_set + np.arange(k, k + 1) for k in range(
            #     self.n_test_samples)])

            next_state_test_indices = test_indices + 1
            x_test = true_data[state_cols + input_cols + disturbance_cols].iloc[test_indices].values
            y_prior_test = 0 if zero_prior else true_data[output_state_col].iloc[test_indices].values
            y_true = true_data[output_state_col].iloc[next_state_test_indices].values - y_prior_test

        else:

            training_indices = np.arange(n_training_samples)
            test_indices = training_indices + test_offset
                # np.arange(test_offset * func['max_n_training_samples'] + 1,
                #                      test_offset * func['max_n_training_samples'] + 1 + self.n_test_samples)
            next_state_test_indices = test_indices + 1
            # test_indices = None
            x_test = true_data[state_cols + input_cols + disturbance_cols].iloc[test_indices].values
            y_prior_test = 0 if zero_prior else true_data[output_state_col].iloc[test_indices].values
            y_true = true_data[output_state_col].iloc[next_state_test_indices].values - y_prior_test

        next_state_training_indices = training_indices + 1
        # for i in range(len(training_indices)):
        #     while not true_data.iloc[training_indices[i]]['isgood']:
        #         training_indices[i] = training_indices[i] + 1
        #
        # for i in range(len(test_indices)):
        #     while not true_data.iloc[test_indices[i]]['isgood']:
        #         test_indices[i] = test_indices[i] + 1

        x_train = true_data[state_cols + input_cols + disturbance_cols].iloc[training_indices].values
        y_prior_train = 0 if zero_prior else true_data[output_state_col].iloc[training_indices].values
        y_train = true_data[output_state_col].iloc[next_state_training_indices].values - y_prior_train

        return training_indices, x_train, y_train, test_indices, x_test, y_true

    def generate_training_data(self, device, func, mpc_t_step, y_true_func, y_prior_func, is_state_gp, output_dim,
                               run_mpc):
        n_training_samples = func['n_init_training_samples'] if run_mpc else func['max_n_training_samples']
                             # + int(self.n_simulation_steps * (mpc_t_step / func['sampling_t_step'])) \

        if is_state_gp:
            x_train = np.hstack([np.random.uniform(device.numerical_bounds[l][0], device.numerical_bounds[l][1],
                                                   (n_training_samples, 1))
                                 for d in range(self.state_lag + 1) for l in device.states] +
                                [np.random.uniform(device.numerical_bounds[l][0], device.numerical_bounds[l][1],
                                                   (n_training_samples, 1))
                                 for d in range(self.input_lag + 1) for l in device.inputs] +
                                [np.random.uniform(device.numerical_bounds[l][0], device.numerical_bounds[l][1],
                                                   (n_training_samples, 1))
                                 for d in range(self.disturbance_lag + 1) for l in device.disturbances])
        else:
            x_train = np.hstack([np.random.uniform(device.numerical_bounds[l][0], device.numerical_bounds[l][1],
                                                   (n_training_samples, 1)) for l in device.states] +
                                [np.random.uniform(device.numerical_bounds[l][0], device.numerical_bounds[l][1],
                                                   (n_training_samples, 1)) for l in device.inputs] +
                                [np.random.uniform(device.numerical_bounds[l][0], device.numerical_bounds[l][1],
                                                   (n_training_samples, 1)) for l in device.disturbances]
                                )

        # generate true next state values
        y_true = []
        for n in range(n_training_samples):
            device.set_simulation_step(int(n * func['sampling_t_step'] / mpc_t_step) % self.n_simulation_steps)
            y_true_sample = np.array(y_true_func(x_train[n]))
            # y_true.append(y_true_sample[output_dim] if y_true_sample.ndim else y_true_sample)
            y_true.append(y_true_sample[output_dim])

        # multidim = y_true_sample.ndim

        y_true = np.vstack(y_true)

        # generate state feedback measurement noise
        noise_train = np.random.normal(0, func['meas_noise'] * np.max(np.abs(y_true)), (n_training_samples, 1))
        n_x_train_vars = x_train.shape[1]
        noise_train_ind = np.random.normal(0, func['meas_noise'] * np.max(np.abs(y_true)),
                                           (n_training_samples * n_x_train_vars, 1))

        # generate next state output training data
        y_train = y_true + noise_train

        x_train_ind = np.zeros((n_x_train_vars * n_training_samples, n_x_train_vars))

        for c in range(n_x_train_vars):
            x_train_ind[c * n_training_samples:(c + 1) * n_training_samples, c] = x_train[:, c]

        y_train_ind = []
        for n in range(n_training_samples * n_x_train_vars):
            y_train_sample = np.array(y_true_func(x_train_ind[n]) + noise_train_ind[n])
            # y_train_ind.append(y_train_sample[output_dim] if y_train_sample.ndim else y_train_sample)
            y_train_ind.append(y_train_sample[output_dim])

        y_train_ind = np.vstack(y_train_ind)

        # y_prior_train = np.vstack([y_prior_func(x_train[n])[output_dim] if multidim else y_prior_func(x_train[n])
        #                            for n in range(func['max_n_training_samples'])])
        y_prior_train = np.vstack([y_prior_func(x_train[n])[output_dim] for n in range(n_training_samples)])

        y_train = np.vstack([y_train[n] - y_prior_train[n] for n in range(n_training_samples)])

        y_prior_train_ind = np.vstack([y_prior_func(x_train_ind[n])[output_dim] for n in
                                       range(n_training_samples * n_x_train_vars)])

        y_train_ind = np.vstack([y_train_ind[n] - y_prior_train_ind[n]
                                 for n in range(n_training_samples * n_x_train_vars)])

        return x_train, x_train_ind, y_true, y_train, y_train_ind

    def generate_disturbances(self, model, n_simulation_steps, n_horizon, use_linear_test_values=True):

        disturbances = np.zeros((n_simulation_steps + n_horizon, 0))

        for device in model.devices:
            if device.n_disturbances:
                if use_linear_test_values:
                    new_disturbances = np.hstack([np.random.uniform(device.numerical_bounds[w][0],
                                                                               device.numerical_bounds[w][1],
                                                                               (n_simulation_steps + n_horizon, 1))
                                                              for w in device.disturbances])
                else:
                    new_disturbances = np.hstack([np.random.normal(device.numerical_gaussian[w]['mean'],
                                                                           device.numerical_gaussian[w]['std'],
                                                                           (n_simulation_steps + n_horizon, 1))
                                                          for w in device.disturbances])
            else:
                new_disturbances = np.zeros((n_simulation_steps + n_horizon, device.n_disturbances))

            disturbances = np.hstack([disturbances, new_disturbances])

        return disturbances

    def generate_current_data(self, device, n_test_samples, use_linear_test_values=True):

        if use_linear_test_values:
            # generate current test input data
            current_state_test = np.hstack([np.random.uniform(device.numerical_bounds[x][0],
                                                             device.numerical_bounds[x][1],
                                                             (n_test_samples, 1))
                                            for x in device.states])

            current_input_test = np.hstack([np.random.uniform(device.numerical_bounds[u][0],
                                                             device.numerical_bounds[u][1],
                                                             (n_test_samples, 1))
                                            for u in device.inputs])

            if device.n_disturbances:
                current_disturbance_test = np.hstack([np.random.uniform(device.numerical_bounds[w][0],
                                                                       device.numerical_bounds[w][1],
                                                                       (n_test_samples, 1))
                                                      for w in device.disturbances])
            else:
                current_disturbance_test = np.zeros((n_test_samples, device.n_disturbances))
        else:
            # generate current test input data
            current_state_test = np.hstack([np.random.normal(device.numerical_gaussian[x]['mean'],
                                                             device.numerical_gaussian[x]['std'],
                                                             (n_test_samples, 1))
                                            for x in device.states])

            current_input_test = np.hstack([np.random.normal(device.numerical_gaussian[u]['mean'],
                                                             device.numerical_gaussian[u]['std'],
                                                             (n_test_samples, 1))
                                            for u in device.inputs])

            if device.n_disturbances:
                current_disturbance_test = np.hstack([np.random.normal(device.numerical_gaussian[w]['mean'],
                                                                       device.numerical_gaussian[w]['std'],
                                                                       (n_test_samples, 1))
                                                      for w in device.disturbances])
            else:
                current_disturbance_test = np.zeros((n_test_samples, device.n_disturbances))

        np.random.shuffle(current_state_test)
        np.random.shuffle(current_input_test)
        np.random.shuffle(current_disturbance_test)

        return current_state_test, current_input_test, current_disturbance_test

    def generate_lagged_data(self, n_samples, state=None, input=None, disturbance=None):

        if state is not None:

            lagged_output = []
            for s in range(n_samples):
                # we need a state lage of self.state_lag and we have (s-1) available lagged states,
                # find the number of initial zeros we need to fill the unavailable lagged states
                n_available_lagged_states = min(max(0, s - 1), self.state_lag)
                n_missing_lagged_states = max(0, self.state_lag - n_available_lagged_states)
                init_zeros = np.zeros(n_missing_lagged_states * self.n_states)
                available_lagged_states = np.hstack(state[s - n_available_lagged_states:s + 1])

                lagged_output.append(np.hstack([init_zeros, available_lagged_states]))

            lagged_output = np.vstack(lagged_output)
        else:
            lagged_output = None

        if input is not None:

            lagged_input = []
            for s in range(n_samples):
                # we need a state lage of self.state_lag and we have (s-1) available lagged states,
                # find the number of initial zeros we need to fill the unavailable lagged states
                n_available_lagged_inputs = min(max(0, s - 1), self.input_lag)
                n_missing_lagged_inputs = max(0, self.input_lag - n_available_lagged_inputs)
                init_zeros = np.zeros(n_missing_lagged_inputs * self.n_inputs)
                available_lagged_inputs = np.hstack(input[s - n_available_lagged_inputs:s + 1])

                lagged_input.append(np.hstack([init_zeros, available_lagged_inputs]))

            lagged_input = np.vstack(lagged_input)

        else:
            lagged_input = None

        if disturbance is not None:

            lagged_disturbance = []
            for s in range(n_samples):
                # we need a state lage of self.state_lag and we have (s-1) available lagged states,
                # find the number of initial zeros we need to fill the unavailable lagged states
                n_available_lagged_disturbances = min(max(0, s - 1), self.disturbance_lag)
                n_missing_lagged_disturbances = max(0, self.disturbance_lag - n_available_lagged_disturbances)
                init_zeros = np.zeros(n_missing_lagged_disturbances * self.n_disturbances)
                available_lagged_disturbances = np.hstack(disturbance[s - n_available_lagged_disturbances:s + 1])

                lagged_disturbance.append(np.hstack([init_zeros, available_lagged_disturbances]))

            lagged_disturbance = np.vstack(lagged_disturbance)
        else:
            lagged_disturbance = None

        return lagged_output, lagged_input, lagged_disturbance

    def generate_measurements(self, controller, simulator, estimator, x0, n_samples,
                              input_train=None, noise_train=None, disturbance_train=None):

        # generate lagged output, control input and disturbance INPuT training data
        # samples are along 0th axis. For each sample row, all features (dim 1, 2 ...)
        # of all variables (y_(k-1), y_(k-2) ...., u_(k-1), ... u_k, w_(k-1), ..., w_k)
        # lagged_next_state_train = np.random.uniform(lb, ub,
        #                                   (func['func['max_n_training_samples']'], self.state_lag * self.n_states))  # sampled lagged outputs

        np.random.seed(99)
        if input_train is None:
            controller.x0 = x0
            controller.set_initial_guess()

        simulator.x0 = x0
        estimator.x0 = x0
        est_next_state = x0.T
        meas_next_state = x0.T

        for s in range(n_samples):
            # these are the vectors stored in state_train
            est_current_state = est_next_state
            if input_train is None:
                # generate optimal controller
                current_input = controller.make_step(est_next_state)
            else:
                # use given control inputs
                current_input = input_train[s, np.newaxis].T
            meas_next_state = simulator.make_step(u0=current_input, v0=noise_train[s, np.newaxis].T,
                                                  w0=disturbance_train[s, np.newaxis].T)
            est_next_state = estimator.make_step(meas_next_state)

        current_state_train = simulator.data['_x']
        next_state_train = np.vstack([current_state_train[1:], meas_next_state.T])
        cost_train = simulator.data['_aux', 'unknown_stage_cost']
        lagged_output_train, _, _ = self.generate_lagged_data(state=current_state_train)

        return lagged_output_train, current_state_train, next_state_train, cost_train

    def set_system_model(self, unknown_system_model):
        self.unknown_system_model = unknown_system_model

    def set_cost_function(self, known_terminal_cost, unknown_terminal_cost, known_stage_cost, unknown_stage_cost):
        self.known_stage_cost = known_stage_cost
        self.known_terminal_cost = known_terminal_cost
        self.unknown_stage_cost = unknown_stage_cost
        self.unknown_terminal_cost = unknown_terminal_cost

    def add_training_data(self, df, training_indices, x_train, y_train, is_init=False, n_init_training_samples=0):

        new_df = pd.DataFrame({'training_indices': training_indices,
                               'x_train': list(x_train),
                               'y_train': list(y_train),
                               'is_init': [True if i < n_init_training_samples else False for i in training_indices] \
                                   if is_init else [False for i in training_indices]})

        # TODO test
        df = pd.concat([new_df, df])
        return df

    # generate nonlinear dynamic system device OuTPuT training data
    # autoregressive system device: current state depends oh previous outputs, control inputs and disturbances
