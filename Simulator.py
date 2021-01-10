import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


class Simulator:
    def __init__(self, ocp, mpc, model, n_simulation_steps):

        self.ocp = ocp
        self.mpc = mpc
        self.model = model
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
        self.regret_trajectory = None
        self.ave_regret_trajectory = None
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
        self.regret_trajectory = np.zeros((0, 1))
        self.ave_regret_trajectory = np.zeros((0, 1))

    def set_trajectories(self, traj_df):
        self.state_trajectory = traj_df.x0
        #self.output_trajectory = np.zeros((0, self.ocp.n_states))
        self.input_trajectory = traj_df.u0
        self.disturbance_trajectory = traj_df.w0
        self.output_trajectory = traj_df.y0
        self.cost_trajectory = traj_df.cost
        self.ineq_constraint_trajectory = traj_df.ineq_constraints
        self.eq_constraint_trajectory = traj_df.eq_contraints

    def modelled_next_state_func(self, z):
        if self.ocp.model_type == 'continuous':
            next_state = (self.ocp.current_states(z, 0) + self.ocp.cont_calculated_state_changes(z, 0))

        elif self.ocp.model_type == 'discrete':
            next_state = self.ocp.disc_calculated_next_states(z, 0)

        return next_state

    def true_cost_func(self, z):

        stage_cost = np.sum([self.true_stage_cost_func(np.hstack(
            [self.ocp.x_stage(z, k), self.ocp.u_stage(z, k), self.ocp.w_stage(k)])) for k in range(self.ocp.n_horizon)])

        term_cost = self.true_terminal_cost_func(z[self.ocp.next_state_indices[-1]])

        cost = float(stage_cost + term_cost)

        return cost

    # def true_equality_constraint_func(self, z):
    #     true_next_states = self.true_next_state_func(z)
    #     calculated_next_states = z[self.ocp.next_state_indices]
    #     return calculated_next_states - true_next_states

    def simulate_thread_func(self, init_state, true_state, true_disturbances, synthetic_data=False):
        self.simulate(init_state, true_state, true_disturbances, synthetic_data)

    def simulate(self, init_state, true_state, true_disturbances, synthetic_data=False):

        x0 = init_state

        # initialise the lagged states vector with the initial state
        lagged_states = np.concatenate([init_state for k in range(self.ocp.state_lag + 1)])

        # initialise the lagged inputs vector with the initial input
        init_input = np.zeros(self.model.n_inputs)
        lagged_inputs = np.tile(init_input, self.ocp.input_lag + 1)

        # initialise the lagged disturbance vector with the initial disturbance
        init_disturbance = true_disturbances[0] if len(true_disturbances) else None
        lagged_disturbances = np.tile(init_disturbance, self.ocp.disturbance_lag + 1)

        self.mpc.z_init = np.zeros(self.ocp.n_optimization_vars
                                   + self.ocp.n_exp_ineq_constraints
                                   + self.ocp.n_exp_eq_constraints)

        for k0 in range(self.n_simulation_steps):

            self.model.set_simulation_step(k0)
            self.ocp.true_disturbances = true_disturbances[k0:k0 + self.ocp.n_horizon].flatten()

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
            z_opt = np.array(res_opt.x, dtype='float64')
            self.mpc.z_init = z_opt

            # get optimal primal variables
            z = res_opt.x[:self.ocp.n_total_vars]

            ineq_constraint = [con['fun'](z)[:int(self.ocp.n_exp_ineq_constraints / self.ocp.n_horizon)]
                               for con in self.ocp.exp_ineq_constraints]
            eq_constraint = [con['fun'](z)[:int(self.ocp.n_exp_eq_constraints / self.ocp.n_horizon)]
                             for con in self.ocp.exp_eq_constraints]

            self.ineq_constraint_trajectory = np.vstack([self.ineq_constraint_trajectory, ineq_constraint])
            self.eq_constraint_trajectory = np.vstack([self.eq_constraint_trajectory, eq_constraint])

            if k0 == 0:
                self.mpc.opt_object.plot()

            z0 = z[:self.ocp.n_stage_vars]

            # get optimal cost and add to cost trajectory
            modelled_cost = np.array([res_opt.fun])[0]

            # if should_cost != cost[0]:
            #     calc_cost = self.ocp.cost_func(res_opt.x[:self.ocp.n_total_vars])
            #     print(calc_cost)

            # get current (first) optimal input and add to input trajectory
            u0 = z0[:self.ocp.n_inputs]
            self.input_trajectory = np.vstack([self.input_trajectory, u0])
            # add current input to lagged inputs
            lagged_inputs = np.concatenate([lagged_inputs[self.ocp.n_inputs:], u0])

            # plug current state and optimized control input into system and fetch true next state
            # if the output of the dynamic state function is the rate of change, assume continuous rate of change
            # over our time step to find next state

            z_lagged = np.hstack([lagged_states, lagged_inputs, lagged_disturbances])

            true_cost = self.true_cost_func(z)
            self.cost_trajectory = np.vstack([self.cost_trajectory, true_cost])

            # np.sum(list([x0[0] ** 2]) + list(res_opt.x[[x[0] for x in self.ocp.next_state_indices]] ** 2))
            # print(true_cost, modelled_cost)
            # print(x0[0], res_opt.x[[x[0] for x in self.ocp.next_state_indices]])

            modelled_next_state = self.modelled_next_state_func(z)
            if synthetic_data:
                if self.ocp.model_type == 'continuous':
                    x0 = x0 + (self.true_next_state_func(z=z_lagged) * self.ocp.mpc_t_step)
                elif self.ocp.model_type == 'discrete':
                    x0 = self.true_next_state_func(z_lagged=z_lagged)
            else:
                # TODO
                # x0 = true_state[k0 + 1]
                x0 = modelled_next_state #z0[self.ocp.n_inputs:self.ocp.n_inputs + self.ocp.n_states]

            self.regret_trajectory = np.vstack([self.regret_trajectory, self.mpc.opt_object.regret])
            self.ave_regret_trajectory = np.vstack([self.ave_regret_trajectory, self.mpc.opt_object.ave_regret])

            print(f'\nTime-Step == {k0}\n'
                  f'primal variables == {z}\n'
                  f'dual variables == {res_opt.x[self.ocp.n_total_vars:]}\n'
                  f'modelled cost == {modelled_cost}\n'
                  f'true cost == {true_cost}\n'
                  f'modelled next state == {modelled_next_state}\n'
                  f'true next state == {x0}\n')

            # TODO add state estimator?

        return self.state_trajectory, self.output_trajectory, self.input_trajectory, self.disturbance_trajectory, \
               self.cost_trajectory,  self.regret_trajectory, self.ave_regret_trajectory, \
               self.ineq_constraint_trajectory, self.eq_constraint_trajectory

    def plot_trajectory(self, comp_sim_df=None):

        # plot for states, inputs, cost, regret ineq constraint violation, eq constraint violation
        n_plots = 7
        self.traj_fig, self.traj_ax = plt.subplots(n_plots, sharex=True, frameon=False)
        self.traj_fig.align_ylabels()
        self.traj_ax[-1].set_xlabel('time-step')

        self.traj_ax[0].set_ylabel('$x_0$', rotation=0)
        self.traj_ax[1].set_ylabel('$u^*_0$', rotation=0)

        self.traj_ax[2].set_ylabel('cost', rotation=0)
        self.traj_ax[3].set_ylabel('regret', rotation=0)
        self.traj_ax[4].set_ylabel('average regret', rotation=0)

        self.traj_ax[5].set_ylabel('ineq constraint', rotation=0)
        self.traj_ax[6].set_ylabel('eq constraint', rotation=0)

        xticks = np.linspace(0, self.n_simulation_steps, 11) if self.n_simulation_steps >= 20 \
            else np.arange(self.n_simulation_steps)

        time_series = np.linspace(0, self.n_simulation_steps - 1, num=self.n_simulation_steps)

        for x in range(self.ocp.n_states):
            self.traj_ax[0].plot(time_series, self.state_trajectory[:, x], label=f"$x_{{0,{x}}}$")
            # if an online trajectory, also plot the online verions
            if comp_sim_df is not None:
                self.traj_ax[0].plot(time_series, np.vstack(comp_sim_df.x0.values)[:, x],
                                     # label=f"Offline $x_{{0,{x}}}$",
                                     linestyle='dashed', color=self.traj_ax[0].get_lines()[-1].get_color())

        for u in range(self.ocp.n_inputs):
            self.traj_ax[1].plot(time_series, self.input_trajectory[:, u], label=f"$u_{{0,{u}}}$")
            if comp_sim_df is not None:
                self.traj_ax[1].plot(time_series, np.vstack(comp_sim_df.u0.values)[:, u],
                                     linestyle='dashed', color=self.traj_ax[1].get_lines()[-1].get_color())

        self.traj_ax[2].plot(time_series, self.cost_trajectory)
        if comp_sim_df is not None:
            self.traj_ax[2].plot(time_series, np.vstack(comp_sim_df.cost.values)[:, 0],
                                 linestyle='dashed', color=self.traj_ax[2].get_lines()[-1].get_color())

        self.traj_ax[3].plot(time_series, self.regret_trajectory)
        if comp_sim_df is not None:
            self.traj_ax[3].plot(time_series, np.vstack(comp_sim_df.regret.values)[:, 0],
                                 linestyle='dashed', color=self.traj_ax[3].get_lines()[-1].get_color())

        self.traj_ax[4].plot(time_series, self.ave_regret_trajectory)
        if comp_sim_df is not None:
            self.traj_ax[4].plot(time_series, np.vstack(comp_sim_df.average_regret.values)[:, 0],
                                 linestyle='dashed', color=self.traj_ax[4].get_lines()[-1].get_color())

        for c in range(int(self.ocp.n_exp_ineq_constraints / self.ocp.n_horizon)):
            self.traj_ax[5].plot(time_series, self.ineq_constraint_trajectory[:, c], label=f"$g_{{{c}}}$",)
            if comp_sim_df is not None:
                self.traj_ax[5].plot(time_series, np.vstack(comp_sim_df.ineq_constraints.values)[:, c],
                                     linestyle='dashed', color=self.traj_ax[5].get_lines()[-1].get_color())

        for c in range(int(self.ocp.n_exp_eq_constraints / self.ocp.n_horizon)):
            self.traj_ax[6].plot(time_series, self.eq_constraint_trajectory[:, c], label=f"$F_{{{c}}}$")
            if comp_sim_df is not None:
                self.traj_ax[6].plot(time_series, np.vstack(comp_sim_df.eq_constraints.values)[:, c],
                                     linestyle='dashed', color=self.traj_ax[6].get_lines()[-1].get_color())

        for traj_ax in self.traj_ax:
            traj_ax.set_xticks(xticks)
            traj_ax.spines['top'].set_visible(False)
            traj_ax.spines['right'].set_visible(False)
            if comp_sim_df is not None:
                handles, labels = traj_ax.get_legend_handles_labels()
                solid_line = traj_ax.plot(time_series, np.zeros_like(time_series), color='black', linestyle='dashed')
                handles = handles + solid_line
                labels = labels + ['Offline']
                traj_ax.legend(handles, labels, bbox_to_anchor=(1, 1), ncol=3)
                solid_line[0].set_visible(False)
            else:
                traj_ax.legend(bbox_to_anchor=(1, 1), ncol=3)

        self.traj_fig.show()

        return self.traj_fig, self.traj_ax

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
