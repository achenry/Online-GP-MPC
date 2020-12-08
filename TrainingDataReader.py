import numpy as np
import pandas as pd


class TrainingDataReader:
    """
    take noisy measurements from actual functions or take infeed of measurements from external sources
    output feed of training data
    """

    def __init__(self, input_params, n_states, n_inputs, n_disturbances):
        self.n_training_samples = input_params['n_training_samples']
        self.n_test_samples = input_params['n_test_samples']
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_disturbances = n_disturbances
        self.state_lag = input_params['state_lag']
        self.input_lag = input_params['input_lag']
        self.disturbance_lag = input_params['disturbance_lag']
        self.n_horizon = input_params['n_horizon']
        self.unknown_system_model = None
        self.known_stage_cost = None
        self.known_terminal_cost = None
        self.unknown_stage_cost = None
        self.unknown_terminal_cost = None

    def generate_next_state_training_data(self, states, inputs, disturbances, bounds):

        # state_x_train_grid = [np.linspace(bounds[l][0], bounds[l][1], self.n_training_samples)
        #                       for d in range(self.state_lag + 1) for l in states] + \
        #                      [np.linspace(bounds[l][0], bounds[l][1], self.n_training_samples)
        #                       for d in range(self.input_lag + 1) for l in inputs] + \
        #                      [np.linspace(bounds[l][0], bounds[l][1], self.n_training_samples)
        #                       for d in range(self.disturbance_lag + 1) for l in disturbances]
        #
        # # returns n_states matrices, indexed by (i,j,k,l,..)
        # state_x_train_mgrid = np.meshgrid(*state_x_train_grid, indexing='ij')
        state_x_train_mgrid = None

        # state_n_dims = (self.n_states * (self.state_lag + 1)) \
        #                + (self.n_inputs * (self.input_lag + 1)) \
        #                + (self.n_disturbances * (self.disturbance_lag + 1))

        state_n_samples = self.n_training_samples  # n_training_samples ** state_n_dims
        # state_x_train = np.zeros((state_n_samples, state_n_dims))


        state_x_train = np.hstack([np.random.uniform(bounds[l][0], bounds[l][1], (self.n_training_samples, 1))
                                  for d in range(self.state_lag + 1) for l in states] +
                                 [np.random.uniform(bounds[l][0], bounds[l][1], (self.n_training_samples, 1))
                                  for d in range(self.input_lag + 1) for l in inputs] +
                                  [np.random.uniform(bounds[l][0], bounds[l][1], (self.n_training_samples, 1))
                                   for d in range(self.disturbance_lag + 1) for l in disturbances]
                                  )

        # for i in np.arange(state_n_samples):
        #     # index = i
        #     # indices = []
        #     # for d in np.arange(state_n_dims):
        #     #    indices.append(int(index % n_training_samples))
        #     #    index /= n_training_samples
        #     indices = [int(np.random.uniform(0, state_n_samples)) for d in range(state_n_dims)]
        #     state_x_train[i] = np.hstack([state_x_train_mgrid[d][tuple(indices)] for d in range(state_n_dims)])

        return state_x_train_mgrid, state_x_train

    def generate_cost_training_data(self, states, inputs, bounds):

        # cost_x_train_grid = [np.linspace(bounds[l][0], bounds[l][1], self.n_training_samples)
        #                      for l in states] + \
        #                     [np.linspace(bounds[l][0], bounds[l][1], self.n_training_samples)
        #                      for l in inputs]
        #
        # cost_x_train_mgrid = np.meshgrid(*cost_x_train_grid, indexing='ij')
        cost_x_train_mgrid = None

        cost_n_dims = self.n_states + self.n_inputs
        cost_n_samples = self.n_training_samples  # n_training_samples ** cost_n_dims
        # cost_x_train = np.zeros((cost_n_samples, cost_n_dims))

        cost_x_train = np.hstack([np.random.uniform(bounds[l][0], bounds[l][1], (self.n_training_samples, 1))
                                  for l in states] +
                                 [np.random.uniform(bounds[l][0], bounds[l][1], (self.n_training_samples, 1))
                                  for l in inputs])

        #for s in range(cost_n_samples):
            # index = i
            # indices = []
            # for d in np.arange(cost_n_dims):
            #    indices.append(int(index % n_training_samples))
            #    index /= n_training_samples

            # indices = [int(np.random.uniform(0, cost_n_samples)) for d in range(cost_n_dims)]
            # cost_x_train[s] = np.hstack([cost_x_train_mgrid[d][tuple(indices)] for d in range(cost_n_dims)])



        return cost_x_train_mgrid, cost_x_train

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
        #                                   (self.n_training_samples, self.state_lag * self.n_states))  # sampled lagged outputs

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

    def add_training_data(self, df, x_train, y_train):

        new_df = pd.DataFrame({'x_train': list(x_train),
                               'y_train': list(y_train)})

        df = df.append(new_df)
        return df

    # generate nonlinear dynamic system model OuTPuT training data
    # autoregressive system model: current state depends oh previous outputs, control inputs and disturbances
