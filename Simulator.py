import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Bbox
import pandas as pd


class Simulator:
    def __init__(self, ocp, mpc, model, data_reader, n_simulation_steps):

        self.ocp = ocp
        self.mpc = mpc
        self.model = model
        self.data_reader = data_reader
        self.n_simulation_steps = n_simulation_steps

        self.true_next_state_func = model.true_next_state_func
        self.true_stage_cost_func = model.true_stage_cost_func
        self.true_terminal_cost_func = model.terminal_cost_func

        self.state_trajectory = None
        self.output_trajectory = None
        self.input_trajectory = None
        self.disturbance_trajectory = None
        self.output_trajectory = None
        self.cost_trajectory = None
        self.ineq_constraint_trajectory = None
        self.eq_constraint_trajectory = None
        self.ineq_dual_trajectory = None
        self.eq_dual_trajectory = None
        self.regret_trajectory = None
        self.ave_regret_trajectory = None
        self.modelled_state_trajectory = None
        self.stage_cost_trajectory = None
        self.modelled_stage_cost_trajectory = None
        self.reset_trajectories()

        self.numerical_bounds = model.numerical_bounds

    def reset_trajectories(self):
        self.state_trajectory = np.zeros((0, self.ocp.n_states))
        self.output_trajectory = np.zeros((0, self.ocp.n_states))
        self.input_trajectory = np.zeros((0, self.ocp.n_inputs))
        self.disturbance_trajectory = np.zeros((0, self.ocp.n_disturbances))
        self.output_trajectory = np.zeros((0, self.ocp.n_outputs))
        self.cost_trajectory = np.zeros((0, 1))
        self.ineq_constraint_trajectory = np.zeros((0, int(self.ocp.n_exp_ineq_constraints / self.ocp.n_horizon)))
        self.eq_constraint_trajectory = np.zeros((0, int(self.ocp.n_exp_eq_constraints / self.ocp.n_horizon)))

        self.ineq_dual_trajectory = np.zeros((0, self.ocp.n_exp_ineq_constraints))
        self.eq_dual_trajectory = np.zeros((0, self.ocp.n_exp_eq_constraints))

        self.regret_trajectory = np.zeros((0, 1))
        self.ave_regret_trajectory = np.zeros((0, 1))

        self.modelled_state_trajectory = []
        self.stage_cost_trajectory = np.zeros((0, len(self.model.devices)))
        self.modelled_stage_cost_trajectory = []

    def set_trajectories(self, traj_df):
        self.state_trajectory = np.vstack(traj_df.x0.values)
        # self.output_trajectory = np.zeros((0, self.ocp.n_states))
        self.input_trajectory = np.vstack(traj_df.u0.values)
        self.disturbance_trajectory = np.vstack(traj_df.w0.values)
        self.cost_trajectory = np.vstack(traj_df.cost.values)
        self.ineq_constraint_trajectory = np.vstack(traj_df.ineq_constraints.values)
        self.eq_constraint_trajectory = np.vstack(traj_df.eq_constraints.values)
        self.ineq_dual_trajectory = np.vstack(traj_df.ineq_dual_vars.values)
        self.eq_dual_trajectory = np.vstack(traj_df.eq_dual_vars.values)

    def modelled_next_state_func(self, z_tau, k0):
        if self.ocp.model_type == 'continuous':
            next_state = (self.ocp.current_states(z_tau, 0) + self.ocp.cont_calculated_state_changes(z_tau, 0))

        elif self.ocp.model_type == 'discrete':
            next_state, next_state_std = self.ocp.disc_calculated_next_states(z_tau, 0, k0, return_std=True)

        return next_state, next_state_std

    def modelled_stage_cost_func(self, z_lagged, k0):
        stage_cost, stage_cost_std = self.ocp.device_stage_cost_func(z_lagged, k0, return_std=True)

        return stage_cost, stage_cost_std

    def true_cost_func(self, z_tau, k0):

        stage_cost = 0
        for k in range(self.ocp.n_horizon):
            self.model.set_simulation_step(k0 + k)
            stage_cost = stage_cost + self.true_stage_cost_func(np.hstack(
                [self.ocp.x_stage(z_tau, k), self.ocp.u_stage(z_tau, k), self.ocp.w_stage(k)]))

        self.model.set_simulation_step(k0 + self.ocp.n_horizon)
        term_cost = self.true_terminal_cost_func(z_tau[self.ocp.next_state_indices[-1]])

        horizon_cost = float(stage_cost + term_cost)

        return horizon_cost

    # def true_equality_constraint_func(self, z):
    #     true_next_states = self.true_next_state_func(z)
    #     calculated_next_states = z[self.ocp.next_state_indices]
    #     return calculated_next_states - true_next_states

    def simulate_thread_func(self, init_state, true_state, true_disturbances,
                             is_simulation_done, is_stage_cost_gp_done, is_next_state_gp_done,
                             synthetic_data=False):

        self.simulate(init_state, true_state, true_disturbances, synthetic_data,
                      is_simulation_done, is_stage_cost_gp_done, is_next_state_gp_done)

    def update_gps(self, stage_cost_gps, stage_cost_training_data_dfs, next_state_gps, next_state_training_data_dfs,
                   k0):
        # update gps if sampling time has passed
        for g, gp in enumerate(stage_cost_gps):
            if gp is not None:
                delta_t = (k0 - gp.sampling_clock) * self.ocp.mpc_t_step  # time.time() - gp.sampling_clock
                if gp is not None and delta_t >= gp.sampling_t_step:
                    stage_cost_training_data_dfs[g] = pd.DataFrame(
                        gp.collect_training_data(stage_cost_training_data_dfs[g], int(delta_t / gp.sampling_t_step),
                    is_init=False))
                    gp.sampling_clock = k0  # time.time()
                    # gp.update_device_bounds()
                    # print(f'\nDevice {gp.device.idx} bounds = {[val[k0] for val in gp.device.parameters.values()]}')
                    gp.update_inv_cov_train()

        for g, gp in enumerate(next_state_gps):
            if gp is not None:
                delta_t = (k0 - gp.sampling_clock) * self.ocp.mpc_t_step  # time.time() - gp.sampling_clock
                if gp is not None and delta_t >= gp.sampling_t_step:
                    next_state_training_data_dfs[g] = pd.DataFrame(
                        gp.collect_training_data(next_state_training_data_dfs[g], int(delta_t / gp.sampling_t_step),
                    is_init=False))
                    gp.sampling_clock = k0  # time.time()
                    gp.update_device_bounds(k0, self.ocp.n_horizon)
                    # bounds = np.asarray(list(gp.device.parameters.values()))
                    for b, bound in enumerate(gp.device.parameters):
                        if 'min' in bound or 'max' in bound:
                            pass
                            # print(f'\nDevice {gp.device.idx} {bound} bound = {bounds[b, k0:k0 + self.ocp.n_horizon]}')
                    # gp.calculate_opt_hyperparams()
                    gp.update_inv_cov_train()

    def simulate(self, init_state, true_state, true_disturbances,
                 stage_cost_gps, stage_cost_training_data_dfs,
                 next_state_gps, next_state_training_data_dfs,
                 true_next_state_gps=[],
                 synthetic_state_data=False, is_simulation_done=None, is_stage_cost_gp_done=None,
                 is_next_state_gp_done=None):

        x0 = init_state

        # initialise the lagged states vector with the initial state
        lagged_states = np.concatenate([init_state for k in range(self.ocp.state_lag + 1)])

        # initialise the lagged inputs vector with the initial input
        init_input = np.zeros(self.model.n_inputs)
        lagged_inputs = np.tile(init_input, self.ocp.input_lag + 1)

        # initialise the lagged disturbance vector with the initial disturbance
        init_disturbance = true_disturbances[0] if len(true_disturbances) else None
        lagged_disturbances = np.tile(init_disturbance, self.ocp.disturbance_lag + 1)

        self.mpc.q_init = np.zeros(self.ocp.n_optimization_vars
                                   + self.ocp.n_exp_ineq_constraints
                                   + self.ocp.n_exp_eq_constraints)

        for k in range(self.ocp.n_horizon):
            self.mpc.q_init[k * (self.ocp.n_inputs + self.ocp.n_states) + self.ocp.n_inputs:
                            (k + 1) * (self.ocp.n_inputs + self.ocp.n_states)] = self.model.init_state

        for gp in stage_cost_gps + next_state_gps:
            if gp is not None:
                gp.sampling_clock = 0  # time.time()

        for k0 in range(self.n_simulation_steps):

            self.model.set_simulation_step(k0)
            self.update_gps(stage_cost_gps, stage_cost_training_data_dfs,
                            next_state_gps, next_state_training_data_dfs,
                            k0)

            self.ocp.true_disturbances = np.concatenate([true_disturbances[k0] for i in range(self.ocp.n_horizon)])
                                                         # np.zeros((self.ocp.n_horizon - 1) * self.ocp.n_disturbances)]) #true_disturbances[k0:k0 + self.ocp.n_horizon].flatten()
            # TODO change to ,
            #   because we don't know future disturbances - do we assume zero or equal to current disturbances?


            # get lagged states and add on most recent state
            lagged_states = np.concatenate([lagged_states[self.ocp.n_states:], x0])

            # add current state to trajectory
            self.state_trajectory = np.vstack([self.state_trajectory, x0])

            if self.ocp.n_disturbances:
                # fetch current disturbances from known exogeneous disturbances
                w0 = true_disturbances[k0]
                # add current disturbances to lagged disturbances
                lagged_disturbances = np.concatenate([lagged_disturbances[self.ocp.n_disturbances:], w0])
            else:
                w0 = []
            self.disturbance_trajectory = np.vstack([self.disturbance_trajectory, w0])

            # run mpc optimize
            self.ocp.set_lagged_vars(lagged_states, lagged_inputs, lagged_disturbances)

            res_opt = self.mpc.optimize_horizon(k0=k0)

            q_opt = np.array(res_opt.x, dtype='float64')

            # get optimal primal variables
            z_opt = q_opt[:self.ocp.n_optimization_vars]
            lambda_opt = q_opt[self.ocp.n_optimization_vars:
                               self.ocp.n_optimization_vars + self.ocp.n_exp_ineq_constraints]
            mu_opt = q_opt[self.ocp.n_optimization_vars + self.ocp.n_exp_ineq_constraints:]

            # time-shifted warm start
            self.mpc.q_init = np.concatenate([z_opt[self.ocp.n_inputs + self.ocp.n_states:],
                                              z_opt[-(self.ocp.n_inputs + self.ocp.n_states):],
                                              lambda_opt[int(self.ocp.n_exp_ineq_constraints / self.ocp.n_horizon):],
                                              lambda_opt[-int(self.ocp.n_exp_ineq_constraints / self.ocp.n_horizon):],
                                              mu_opt[int(self.ocp.n_exp_eq_constraints / self.ocp.n_horizon):],
                                              mu_opt[-int(self.ocp.n_exp_eq_constraints / self.ocp.n_horizon):]])

            ineq_constraint = [con['fun'](z_opt, k0)[:int(self.ocp.n_exp_ineq_constraints / self.ocp.n_horizon)]
                               for con in self.ocp.exp_ineq_constraints]
            # eq_constraint = [con['fun'](z_opt, k0)[:int(self.ocp.n_exp_eq_constraints / self.ocp.n_horizon)]
            #                  for con in self.ocp.exp_eq_constraints]

            self.ineq_constraint_trajectory = np.vstack([self.ineq_constraint_trajectory, ineq_constraint])
            self.ineq_dual_trajectory = np.vstack([self.ineq_dual_trajectory, lambda_opt])
            self.eq_dual_trajectory = np.vstack([self.eq_dual_trajectory, mu_opt])

            if k0 == 0:
                self.mpc.opt_object.plot()

            z0 = z_opt[:self.ocp.n_inputs + self.ocp.n_states]

            # get optimal cost and add to cost trajectory
            modelled_cost = np.array([res_opt.fun])[0]

            # get current (first) optimal input and add to input trajectory
            u0 = z0[:self.ocp.n_inputs]
            self.input_trajectory = np.vstack([self.input_trajectory, u0])
            # add current input to lagged inputs
            lagged_inputs = np.concatenate([lagged_inputs[self.ocp.n_inputs:], u0])

            # plug current state and optimized control input into system and fetch true next state
            # if the output of the dynamic state function is the rate of change, assume continuous rate of change
            # over our time step to find next state

            z_lagged = np.hstack([lagged_states, lagged_inputs, lagged_disturbances])

            true_cost = self.true_cost_func(z_opt, k0)
            self.cost_trajectory = np.vstack([self.cost_trajectory, true_cost])

            true_stage_cost = np.concatenate([dev.true_stage_cost_func(z_lagged[self.model.device_zstage_indices[dev.idx]])
                                                        for dev in self.model.devices])
            self.stage_cost_trajectory = np.vstack([self.stage_cost_trajectory, true_stage_cost])

            # modelled_stage_cost_pred = [gp.predict(z_lagged[self.model.device_zstage_indices[gp.device.idx]])
            #                                                      for gp in stage_cost_gps if gp is not None]
            # modelled_stage_cost = [c[0][0][0] for c in modelled_stage_cost_pred]
            # modelled_stage_cost_std = [c[1][0][0] for c in modelled_stage_cost_pred]
            modelled_stage_cost, modelled_stage_cost_std = self.modelled_stage_cost_func(z_lagged, k0)

            self.modelled_stage_cost_trajectory.append((modelled_stage_cost, modelled_stage_cost_std))

            init_state = np.array(x0)

            x0, x0_std = self.true_next_state_func(z_lagged=z_lagged, k=k0, is_synthetic_data=synthetic_state_data,
                                                   return_std=True)
            modelled_next_state, modelled_next_state_std = self.modelled_next_state_func(z_opt, k0)
            self.modelled_state_trajectory.append((modelled_next_state, modelled_next_state_std))

            eq_constraint = (x0 - self.ocp.next_state_func(z_lagged))
            self.eq_constraint_trajectory = np.vstack([self.eq_constraint_trajectory, eq_constraint])

            # TODO
            # update_x0 = False
            # for g, gp in enumerate(true_next_state_gps):
            #     if gp is not None:
            #         d = self.model.devices.index(gp.device)
            #         dim = self.model.device_xstage_indices[d]
            #         if x0_std[dim] >= 1:
            #             gp.prior_mean = gp.device.ref_state[gp.dim]  # + self.state_trajectory[-1, dim]) / 2
            #             update_x0 = True
            #         gp.set_kernel(gp.device.ref_state[gp.dim])
            #
            # if update_x0:
            #     x0 = self.true_next_state_func(z_lagged=z_lagged, k=k0,
            #                                    is_synthetic_data=synthetic_state_data)

            next_state_gp_scores = []
            for g, gp in enumerate(next_state_gps):
                if gp is not None:
                    if True or modelled_next_state_std[self.model.device_xstage_indices[gp.device.idx]][gp.dim] > 1e-3:
                        #     np.max([s[1][self.model.device_xstage_indices[gp.device.idx]][gp.dim]
                        #             for s in self.modelled_state_trajectory]): #1e-4:
                        # print(f'new state std = '
                        #       f'{modelled_next_state_std[self.model.device_xstage_indices[gp.device.idx]][gp.dim]}\n')
                        if synthetic_state_data:
                            next_state_training_data_dfs[g] = \
                                pd.DataFrame(self.data_reader.add_training_data(
                                    next_state_training_data_dfs[g],
                                    np.arange(len(next_state_training_data_dfs[g].index),
                                              len(next_state_training_data_dfs[g].index) + 1).astype('int'),
                                    [z_lagged[self.model.device_zstage_indices[gp.device.idx]]], [x0[g:g + 1]],
                                    is_init=False))
                        else:
                            pass

                        if k0 > 0:
                            indices = self.model.device_xstage_indices[gp.device.idx][gp.dim:gp.dim + 1]
                            next_state_gp_scores = next_state_gp_scores + list(gp.score(
                                y_true=self.state_trajectory[1:, indices],
                                y_pred=np.vstack([s[0][indices] for s in self.modelled_state_trajectory[:-1]])))

            stage_cost_gp_scores = []
            for g, gp in enumerate(stage_cost_gps):
                if gp is not None:
                    if True or modelled_stage_cost_std[gp.device.idx] > 1e-3:
                            # np.max([c[1][gp.device.idx] for c in self.modelled_stage_cost_trajectory]): #1e-4:
                        # print(f'new cost std = '
                        #       f'{modelled_stage_cost_std[self.model.device_xstage_indices[gp.device.idx]][gp.dim]}\n')

                        stage_cost_training_data_dfs[g] = \
                            pd.DataFrame(self.data_reader.add_training_data(
                                stage_cost_training_data_dfs[g],
                                np.arange(len(stage_cost_training_data_dfs[g].index),
                                          len(stage_cost_training_data_dfs[g].index) + 1).astype('int'),
                                [z_lagged[self.model.device_zstage_indices[gp.device.idx]]], [true_stage_cost[g:g+1]],
                                is_init=False))

                        stage_cost_gp_scores = stage_cost_gp_scores + list(gp.score(
                            y_true=self.stage_cost_trajectory[:, g:g+1],
                            y_pred=np.vstack([c[0][g:g+1] for c in self.modelled_stage_cost_trajectory])))

            for gp in true_next_state_gps:
                if gp is not None:
                    gp.prior_mean = 0
                # gp.set_kernel(0)

            # for g, gp in enumerate(true_next_state_gps):
            #
            # for i, x in enumerate(x0):
            #     if modelled_next_state_stdev[i] > 1:
            #         if x < dev.ref_state[]
            #         x0[i] = x + modelled_next_state_stdev[i]

            # if synthetic_state_data:
            #     if self.ocp.model_type == 'continuous':
            #         x0 = x0 + (self.true_next_state_func(z=z_lagged) * self.ocp.mpc_t_step)
            #     elif self.ocp.model_type == 'discrete':
            #         x0 = self.true_next_state_func(z_lagged=z_lagged)
            # else:
            #
            #     # x0 = true_state[k0 + 1] TODO change var nonzero prior
            #     x0 = modelled_next_state
            # for s, gp in enumerate(next_state_gps):
            #     if gp is not None:
            #         state = modelled_next_state[s]
            #         min_state = np.min(gp.y_train[:, gp.output_dim])
            #         max_state = np.max(gp.y_train[:, gp.output_dim])
            #         state_var = state - init_state[s] # gp.device.next_state_prior_func(z_lagged)
            #         state_diff = np.diff(gp.y_train[:, gp.output_dim])
            #         max_state_var = np.max(state_diff)
            #         min_state_var = np.min(state_diff)
            #         if (state_var > max_state_var) or (state_var < min_state_var):
            #             state_var = np.max([min_state_var, np.min([max_state_var, state_var])]) # + init_state[s]
            #
            #         state = state + state_var
            #
            #         if (state > max_state) or (state < min_state):
            #             state = np.max([min_state, np.min([state, max_state])])
            #
            #         x0[s] = state

            # x0 = np.max([np.min([modelled_next_state, x0 + 2], axis=0), x0 - 2], axis=0)
            # x0 = z0[self.ocp.n_inputs:self.ocp.n_inputs + self.ocp.n_states]

            self.regret_trajectory = np.vstack([self.regret_trajectory, self.mpc.opt_object.regret])
            self.ave_regret_trajectory = np.vstack([self.ave_regret_trajectory, self.mpc.opt_object.ave_regret])

            print(f'\nTime-Step == {k0}\n'
                  f'primal variables == {z_opt}\n'
                  f'dual variables == {q_opt[self.ocp.n_optimization_vars:]}\n'
                  f'modelled cost == {modelled_cost}\n'
                  f'modelled stage cost == {modelled_stage_cost, modelled_stage_cost_std}\n'
                  f'true cost == {true_cost}\n'
                  f'initial state == {init_state}\n'
                  f'modelled next state == {modelled_next_state, modelled_next_state_std}\n'
                  f'true next state == {x0}\n'
                  f'true next state standard deviation == {x0_std}\n'
                  f'next state gp scores == {next_state_gp_scores}\n'
                  f'stage cost gp scores == {stage_cost_gp_scores}\n'
                  f'disturbances == {w0}')

        return self.state_trajectory, self.output_trajectory, self.input_trajectory, self.disturbance_trajectory, \
               self.cost_trajectory, self.regret_trajectory, self.ave_regret_trajectory, \
               self.ineq_constraint_trajectory, self.eq_constraint_trajectory

    def plot_trajectory(self, comp_sim_df=None, bounds=None, return_tracking_error=False):

        n_xticks = np.min([8, self.n_simulation_steps])
        xticks = np.linspace(1, self.n_simulation_steps, n_xticks).astype(int).tolist()
        # np.arange(start=0, stop=self.n_simulation_steps, step=time_step)

        time_series = np.arange(1, self.n_simulation_steps + 1).tolist()

        # plot for states, inputs, cost, regret ineq constraint violation, eq constraint violation
        n_plots = 5 if self.ocp.n_exp_ineq_constraints else 4
        self.error_fig, self.error_ax = plt.subplots(n_plots, sharex=True, frameon=False)
        self.traj_fig, self.traj_ax = plt.subplots(n_plots, sharex=True, frameon=False)
        self.dual_fig, self.dual_ax = plt.subplots(2, sharex=True, frameon=False)
        self.regret_fig, self.regret_ax = plt.subplots(2, sharex=True, frameon=False)

        self.dual_fig.align_ylabels()
        self.dual_ax[-1].set_xlabel('time-step')
        self.dual_ax[0].set_ylabel('$\mathbf{\lambda}$', rotation=0)
        self.dual_ax[1].set_ylabel('$\mathbf{\mu}$', rotation=0)

        self.traj_fig.align_ylabels()
        self.traj_ax[-1].set_xlabel('time-step')
        self.traj_ax[0].set_ylabel('$\mathbf{x_0}$', rotation=0)
        self.traj_ax[1].set_ylabel('$\mathbf{u_0}$', rotation=0)
        self.traj_ax[2].set_ylabel('L', rotation=0)
        self.traj_ax[3].set_ylabel('F', rotation=0)
        if self.ocp.n_exp_ineq_constraints:
            self.traj_ax[4].set_ylabel('G', rotation=0)

        self.regret_ax[-1].set_xlabel('time-step')
        self.regret_ax[0].set_ylabel('regret', rotation=0)
        self.regret_ax[1].set_ylabel('average regret', rotation=0)

        self.regret_ax[0].plot(time_series, self.regret_trajectory)
        self.regret_ax[1].plot(time_series, self.ave_regret_trajectory)

        if comp_sim_df is not None:
            comp_state_trajectory = np.vstack(comp_sim_df.x0.values)
            comp_input_trajectory = np.vstack(comp_sim_df.u0.values)
            comp_cost_trajectory = np.vstack(comp_sim_df.cost.values)
            comp_eq_constraint_trajectory = np.vstack(comp_sim_df.eq_constraints.values)
            comp_ineq_constraint_trajectory = np.vstack(comp_sim_df.ineq_constraints.values)

            self.error_fig.align_ylabels()
            self.error_ax[-1].set_xlabel('$k_0$')
            self.error_ax[0].set_ylabel(f'$\left \Vert \hat{{\mathbf{{x_0}}}} - \mathbf{{x^\star_0}} \\right \Vert$',
                                        rotation=0)
            self.error_ax[1].set_ylabel('$\left\Vert \hat{{\mathbf{{u_0}}}} - \mathbf{{u^\star_0}} \\right\Vert$',
                                        rotation=0)
            self.error_ax[2].set_ylabel('$\left\Vert \hat{{L}} - L^\star \\right\Vert$', rotation=0)
            # self.error_ax[3].set_ylabel('regret', rotation=0)
            # self.error_ax[4].set_ylabel('average regret', rotation=0)
            self.error_ax[3].set_ylabel(f'$\left\Vert \hat{{F}} - F^\star \\right\Vert$', rotation=0)
            if self.ocp.n_exp_ineq_constraints:
                self.error_ax[4].set_ylabel(f'$\left\Vert \hat{{G}} - G^\star \\right\Vert$', rotation=0)
            self.error_ax[0].plot(time_series,
                                  np.linalg.norm((comp_state_trajectory - self.state_trajectory), 2, axis=1))
            self.error_ax[1].plot(time_series,
                                  np.linalg.norm((comp_input_trajectory - self.input_trajectory), 2, axis=1))
            self.error_ax[2].plot(time_series,
                                  np.linalg.norm((comp_cost_trajectory - self.cost_trajectory)[:, np.newaxis], 2,
                                                 axis=1))
            self.error_ax[3].plot(time_series,
                                  np.linalg.norm((comp_eq_constraint_trajectory - self.eq_constraint_trajectory), 2,
                                                 axis=1))
            if self.ocp.n_exp_ineq_constraints:
                self.error_ax[4].plot(time_series,
                                      np.linalg.norm(
                                          (comp_ineq_constraint_trajectory - self.ineq_constraint_trajectory), 2,
                                          axis=1))

        for x in range(self.ocp.n_states):
            self.traj_ax[0].plot(time_series, self.state_trajectory[:, x], label=f"$\mathbf{{x_{{0,{x}}}}}$",
                                 linestyle='solid')
            # if an online trajectory, also plot the online version
            if comp_sim_df is not None:
                self.traj_ax[0].plot(time_series, comp_state_trajectory[:, x],
                                     linestyle='dashed', color=self.traj_ax[0].get_lines()[-1].get_color())

        for u in range(self.ocp.n_inputs):
            self.traj_ax[1].plot(time_series, self.input_trajectory[:, u], label=f"$\mathbf{{u_{{0,{u}}}}}$",
                                 linestyle='solid')
            if comp_sim_df is not None:
                self.traj_ax[1].plot(time_series, comp_input_trajectory[:, u],
                                     linestyle='dashed', color=self.traj_ax[1].get_lines()[-1].get_color())

        self.traj_ax[2].plot(time_series, self.cost_trajectory, linestyle='solid')
        if comp_sim_df is not None:
            self.traj_ax[2].plot(time_series, comp_cost_trajectory[:, 0],
                                 linestyle='dashed', color=self.traj_ax[2].get_lines()[-1].get_color())

        # self.traj_ax[3].plot(time_series, self.regret_trajectory)
        # if comp_sim_df is not None:
        #     self.traj_ax[3].plot(time_series, np.vstack(comp_sim_df.regret.values)[:, 0],
        #                          linestyle='dashed', color=self.traj_ax[3].get_lines()[-1].get_color())
        #
        # self.traj_ax[4].plot(time_series, self.ave_regret_trajectory)
        # if comp_sim_df is not None:
        #     self.traj_ax[4].plot(time_series, np.vstack(comp_sim_df.average_regret.values)[:, 0],
        #                          linestyle='dashed', color=self.traj_ax[4].get_lines()[-1].get_color())

        for c in range(int(self.ocp.n_exp_eq_constraints / self.ocp.n_horizon)):
            self.traj_ax[3].plot(time_series, self.eq_constraint_trajectory[:, c], label=f"$F_{{{c}}}$",
                                 linestyle='solid')
            if comp_sim_df is not None:
                self.traj_ax[3].plot(time_series, comp_eq_constraint_trajectory[:, c],
                                     linestyle='dashed', color=self.traj_ax[3].get_lines()[-1].get_color())

        if self.ocp.n_exp_ineq_constraints:
            for c in range(int(self.ocp.n_exp_ineq_constraints / self.ocp.n_horizon)):
                self.traj_ax[4].plot(time_series, self.ineq_constraint_trajectory[:, c], label=f"$g_{{{c}}}$",
                                     linestyle='solid')
                if comp_sim_df is not None:
                    self.traj_ax[4].plot(time_series, comp_ineq_constraint_trajectory[:, c],
                                         linestyle='dashed', color=self.traj_ax[4].get_lines()[-1].get_color())

        if self.ocp.n_exp_ineq_constraints:
            for c in range(self.ocp.n_exp_ineq_constraints):
                self.dual_ax[0].plot(time_series, self.ineq_dual_trajectory[:, c])

        for c in range(self.ocp.n_exp_eq_constraints):
            self.dual_ax[1].plot(time_series, self.eq_dual_trajectory[:, c])

        for a, ax in enumerate(np.concatenate([self.traj_ax, self.error_ax, self.dual_ax, self.regret_ax])):
            ax.set_xticks(xticks)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            y_data = [l.get_data()[1] for l in ax.get_lines()]
            if len(y_data):
                ax.set_ylim([np.min(y_data), np.max(y_data)])
            if bounds is not None and a < len(self.traj_ax):
                ax.set_ylim(bounds[a])
            # ax.set_position()

        for ax in self.traj_ax:
            handles, labels = ax.get_legend_handles_labels()
            if comp_sim_df is not None:
                solid_line = ax.plot(time_series, np.zeros_like(time_series), color='black', linestyle='dashed')
                handles = handles + solid_line
                labels = labels + ['Offline']
                if len(labels) <= 8:
                    ax.legend(handles, labels, bbox_to_anchor=(1, 1), ncol=2)
                solid_line[0].set_visible(False)
            elif len(labels) <= 8:
                ax.legend(bbox_to_anchor=(1, 1), ncol=2)

        self.traj_fig.show()
        for ax in self.traj_ax:
            ax.set_position(ax.get_position())
        self.regret_fig.show()
        # self.error_fig.show()
        # self.dual_fig.show()

        return self.traj_fig, self.error_fig, self.dual_fig, self.regret_fig

    def plot_convergence(self):
        opt_object = self.mpc.opt_object
        return opt_object.conv_fig, opt_object.conv_ax, opt_object.var_fig, opt_object.var_ax

    def animate(self):

        anim_fig, anim_ax = plt.subplots(1)
        anim_ax.set_xlim([-4, 4])
        anim_ax.set_ylim([-4, 4])
        anim_ax.grid()

        line, = anim_ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = anim_ax.text(0.05, 0.9, '', transform=anim_ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            x = [0, np.cos((np.pi / 2) - self.state_trajectory[i, 0])]
            y = [0, np.sin((np.pi / 2) - self.state_trajectory[i, 0])]

            line.set_data(x, y)
            time_text.set_text(time_template % (i * self.ocp.mpc_t_step))
            return line, time_text

        anim = animation.FuncAnimation(anim_fig, animate, np.arange(1, self.n_simulation_steps),
                                       interval=1000 * self.ocp.mpc_t_step, blit=True, init_func=init)

        return anim_fig, anim_ax, anim


class SimulationComparator:
    def __init__(self, sim_df_a, sim_df_b):
        self.columns = sim_df_a.columns
        self.comparison_df = pd.merge(sim_df_a, sim_df_b, on=['time_step'], how='outer', suffixes=('_a', '_b'))

    def plot_rel_error(self):
        fig, ax = plt.subplots(1, 1)
        x = self.comparison_df.index

        col_a = self.comparison_df[[col + '_a' for col in self.columns]].values
        col_b = self.comparison_df[[col + '_b' for col in self.columns]].values
        y = np.linalg.norm(col_a - col_b, axis=1) / \
            np.linalg.norm(col_b, axis=1)
        ax.plot(x, y)

        fig.show()

        return fig, ax
