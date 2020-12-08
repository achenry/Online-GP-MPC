import numpy as np
# from InputReader import InputReader
from matplotlib import animation
from GP import CostGP, NextStateGP
from OCP import OCP
from MPC import MPC
from Simulator import Simulator, SimulationComparator
from TrainingDataReader import TrainingDataReader
from OCPFunctions import OCPFunctions
from threading import Thread, Event
import pandas as pd
from PDOptimizationMethod import PDOptimizationMethod
import os
from helper_functions import gradient_func, jacobian_func
import re


def main(input_params):
    ####################################################################################################################

    # READ INPUT PARAMETERS

    # NOTE: if using GP approx, set model_type to discrete and return cont_func from actual_next_state

    state_lag = input_params['state_lag']
    input_lag = input_params['input_lag']
    disturbance_lag = input_params['disturbance_lag']
    n_horizon = input_params['n_horizon']
    t_step = input_params['t_step']
    n_training_samples = input_params['n_training_samples']
    n_init_training_samples = input_params['n_init_training_samples']
    n_test_samples = input_params['n_test_samples']
    n_simulation_steps = input_params['n_simulation_steps']

    next_state_output_variances = input_params['next_state_output_variances']
    next_state_length_scales = input_params['next_state_length_scales']
    next_state_meas_noises = input_params['next_state_meas_noises']
    cost_output_variance = input_params['cost_output_variance']
    cost_length_scale = input_params['cost_length_scale']
    cost_meas_noise = input_params['cost_meas_noise']

    next_state_sampling_period = input_params['next_state_sampling_period']
    cost_sampling_period = input_params['cost_sampling_period']

    opt_params = {key: input_params[key] for key in ['xtol', 'maxiter', 'lambda_prior', 'mu_prior',
                                                     'alpha', 'eta', 'eps']}

    # plot_gp = input_params['plot_gp']
    use_opt_method = input_params['use_opt_method']
    use_cost_gp_approx = input_params['use_cost_gp_approx']
    use_next_state_gp_approx = input_params['use_next_state_gp_approx']
    model_next_state_error = input_params['model_next_state_error']
    model_type = input_params['model_type']
    model_nonlinear_next_state = input_params['model_nonlinear_next_state']
    simulate_system = input_params['simulate_system']
    compare_simulations = input_params['compare_simulations']
    simulation_name = input_params['simulation_name']
    simulation_comparisons = input_params['simulation_comparisons']
    cost_input_labels = input_params['cost_input_labels']
    next_state_input_labels = input_params['next_state_input_labels']
    next_state_output_labels = input_params['next_state_output_labels']
    imp_bounds_only = input_params['imp_bounds_only']

    sim_dir = f'./results/{simulation_name}'
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)
    else:
        for _, _, files in os.walk(sim_dir):
            for file in files:
                os.remove(f'{sim_dir}/{file}')

    export_dir = './results/exports'

    if not os.path.exists('./results/simulation_parameters.csv'):
        simulation_params_df = pd.DataFrame({key: [val] for key, val in input_params.items()})
    else:
        simulation_params_df = pd.read_csv('./results/simulation_parameters.csv', engine='python', index_col=0)
        simulation_params_df = simulation_params_df.append(input_params, ignore_index=True)
        simulation_params_df = simulation_params_df.reindex()

    if not os.path.exists('./results/gp_results.csv'):
        gp_results_df = pd.DataFrame(columns=['Prediction Name', 'Function', 'No. Training Samples', 'Length Scale',
                                              'Output Variance', 'Measurement Noise', 'Score'])
    else:
        gp_results_df = pd.read_csv('./results/gp_results.csv', engine='python', index_col=0)

    simulation_params_df.to_csv('./results/simulation_parameters.csv')
    sim_params_table_dir = './results/exports/simulation_parameters.txt'

    simulation_number = len(open(sim_params_table_dir).readlines()) + 1 if os.path.exists(sim_params_table_dir) else 1

    sim_params_table_row = f'{simulation_number} & ' \
                           f'{"GP Approximation" if any(use_next_state_gp_approx + [use_cost_gp_approx]) else "Known Model"} ' \
                           f'& {opt_params["maxiter"]} ' \
                           f'& {n_simulation_steps} ' \
                           f'& {n_horizon} & ' \
                           f'{opt_params["alpha"]} & ' \
                           f'{opt_params["eta"]} & ' \
                           f'{opt_params["eps"]} \\\\\n'

    sim_params_table_file = open(f'{sim_params_table_dir}', 'a')
    sim_params_table_file.write(sim_params_table_row)
    sim_params_table_file.close()

    # input_params = InputReader(input_params)
    ####################################################################################################################

    # IMPORT SPECIFIC OPTIMAL CONTROL PROBLEM FUNCTIONS
    ocp_functions = OCPFunctions()

    ####################################################################################################################

    # INITIALISE OPTIMAL CONTROL PROBLEM
    ocp = OCP(n_horizon=n_horizon, state_lag=state_lag, input_lag=input_lag, disturbance_lag=disturbance_lag,
              t_step=t_step, imp_bounds_only=imp_bounds_only, model_type=model_type)
    states, inputs, disturbances, parameters, numerical_bounds = ocp_functions.system_variables()
    n_states = len(states)
    n_inputs = len(inputs)
    n_disturbances = len(disturbances)

    ocp.set_opt_vars(n_states, n_inputs, n_disturbances)
    ocp.set_params(parameters)
    ocp.set_state_ineq_func(ocp_functions.state_ineq_constraint_func)
    ocp.set_imp_bounds(ocp_functions.stage_bounds(), ocp_functions.term_bounds())

    ####################################################################################################################

    # INITIALISE ACTUAL UNKNOWN FUNCTIONS
    actual_next_state_func = ocp_functions.actual_next_state(parameters, t_step, n_states, n_inputs, n_disturbances,
                                                             state_lag, input_lag, disturbance_lag,
                                                             model_nonlinear_next_state)
    # unknown stage cost func and terminal cost functions imported directly

    ####################################################################################################################

    # INITIALISE DYNAMIC-STATE (ERROR) & COST GP APPROXIMATIONS

    # define dynamic state function GP approximation
    next_state_gps = []
    next_state_funcs = []
    next_state_jacobians = []
    if any(use_next_state_gp_approx):
        next_state_prior_func = ocp_functions.state_prior(t_step, n_states, n_inputs, n_disturbances,
                                                          state_lag, input_lag, disturbance_lag)

    for s, use_gp in enumerate(use_next_state_gp_approx):
        if use_gp:
            if model_next_state_error:
                next_state_gps.append(NextStateGP(next_state_output_variances[s], next_state_meas_noises[s],
                                                  next_state_length_scales[s], n_training_samples, n_test_samples,
                                                  state_lag, input_lag, disturbance_lag,
                                                  n_states, n_inputs, n_disturbances, s))

                # call conditional probability calculator each time the function is needs
                next_state_funcs.append(
                    lambda z_lagged:
                    next_state_prior_func(z=z_lagged)[s] + next_state_gps[-1].predict(z_lagged[np.newaxis, :])[0][0])


                next_state_jacobians.append(
                    lambda z_lagged:
                    jacobian_func(next_state_prior_func, z_lagged)[s, :] + next_state_gps[-1].mean_jacobian(z_lagged))

            else:
                next_state_gps.append(NextStateGP(next_state_output_variances[s], next_state_meas_noises[s],
                                                  next_state_length_scales[s], n_training_samples, n_test_samples,
                                                  state_lag, input_lag, disturbance_lag,
                                                  n_states, n_inputs, n_disturbances, s))

                next_state_funcs.append(
                    lambda z_lagged: next_state_gps[-1].predict(z_lagged[np.newaxis, :])[0][0])

                next_state_jacobians.append(next_state_gps[-1].mean_jacobian)
                # next_state_jacobian = next_state_gp.mean_jacobian

        else:
            next_state_gps.append(None)

            # bind s variable as a default parameter to function so that it doesn't always refer to last possible state
            def next_state_func(z_lagged, f=s):
                return actual_next_state_func(z_lagged)[f]

            next_state_funcs.append(next_state_func)

            def next_state_jacobian(z_lagged, f=s):
                return jacobian_func(actual_next_state_func, z_lagged)[f, :]

            next_state_jacobians.append(next_state_jacobian)

    if use_cost_gp_approx:
        # define cost function GP approximation
        cost_gp = CostGP(cost_output_variance, cost_meas_noise, cost_length_scale,
                         n_training_samples, n_test_samples, n_states, n_inputs)

        def stage_cost_func(z_stage):
            # first 0 index fetches the mean, second 0 index fetches the first row of the return predictions
            return cost_gp.predict(z_stage[np.newaxis, :])[0][0]

        # use the gp first-order derivative to calculate the gradient
        stage_cost_jacobian = cost_gp.mean_jacobian
    else:
        stage_cost_func = ocp_functions.actual_stage_cost_func

        def stage_cost_jacobian(z_stage):
            return np.vstack([gradient_func(ocp_functions.actual_stage_cost_func, z_stage)])

    ocp.set_cost_jacobians(stage_cost_jacobian, ocp_functions.terminal_cost_jacobian)
    ocp.set_next_state_jacobians(next_state_jacobians)
    ocp.set_state_ineq_jacobian(ocp_functions.state_ineq_constraint_jacobian)

    ####################################################################################################################

    # GENERATE RANDOM TRAINING DATA

    # instantiate a TrainingDataReader object to generate system measurements
    training_data_reader = TrainingDataReader(input_params, n_states, n_inputs, n_disturbances)
    gp_df_cols = ['x_train', 'y_train']

    if any(use_next_state_gp_approx):
        # generate next state input training data
        state_x_train_mgrid, state_x_train = \
            training_data_reader.generate_next_state_training_data(states, inputs, disturbances, numerical_bounds)

        # generate state feedback measurement noise
        state_noise_train = np.array([np.random.normal(0, [n], size=(n_training_samples, 1))
                                      for n in next_state_meas_noises]).reshape((n_training_samples, n_states))

        # generate next state output training data
        next_state_train = np.vstack(
            [np.hstack([val + state_noise_train[s][d] for d, val in enumerate(
                actual_next_state_func(z=state_x_train[s]))])
             for s in range(n_training_samples)])

        if model_next_state_error:
            next_state_prior_train = np.vstack([next_state_prior_func(z=state_x_train[s])
                                                for s in range(n_training_samples)])

            next_state_error_train = \
                np.vstack([next_state_train[s] - next_state_prior_train[s] for s in range(n_training_samples)])

        # add next state (error) training data to database
    next_state_dfs = []
    for s, use_gp in enumerate(use_next_state_gp_approx):
        if use_gp:
            next_state_df = pd.DataFrame(columns=gp_df_cols)

            next_state_dfs.append(training_data_reader.add_training_data(next_state_df, state_x_train,
                                                                         next_state_error_train[:, s][:,
                                                                         np.newaxis] if model_next_state_error
                                                                         else next_state_train[:, s][:, np.newaxis]))
        else:
            next_state_dfs.append(None)

    if use_cost_gp_approx:
        # generate cost input training data
        cost_x_train_mgrid, cost_x_train = training_data_reader.generate_cost_training_data(states, inputs,
                                                                                            numerical_bounds)

        # generate stage cost feedback measurement noise
        cost_noise_train = np.random.multivariate_normal(
            [0], [[cost_meas_noise]], size=(n_training_samples, 1)).reshape((n_training_samples, 1))

        # generate cost output training data
        cost_train = np.vstack([ocp_functions.actual_stage_cost_func(z_stage=cost_x_train[s]) + cost_noise_train[s][0]
                                for s in np.arange(n_training_samples)])

        # add cost training data to database
        cost_df = pd.DataFrame(columns=gp_df_cols)
        cost_df = training_data_reader.add_training_data(cost_df, cost_x_train, cost_train)

    ####################################################################################################################

    # FETCH INITIAL SAMPLES OF DYNAMIC STATE AND COST FUNCTIONS

    # collect function samples from DataFrame and add to GP training set
    for s, use_gp in enumerate(use_next_state_gp_approx):
        if use_gp:
            next_state_gps[s].collect_training_data(next_state_dfs[s], n_init_training_samples)

    if use_cost_gp_approx:
        cost_gp.collect_training_data(cost_df, n_init_training_samples)

    ####################################################################################################################

    # INITIALISE DYNAMIC-STATE AND COST FUNCTIONS IN OCP
    # if using the gp approximation
    # set the discrete next state to the gp approximation call
    # else if using the actual identified dynamic state and cost functions
    # set the discrete next state to the actual next state function

    ocp.set_next_state_funcs(next_state_funcs)

    # if using gp approximaion
    # set the stage cost to the gp approximation call and the terminal cost the given function
    ocp.set_cost_funcs(stage_cost_func, ocp_functions.actual_terminal_cost_func)

    ####################################################################################################################

    # INITIALISE EXPLICIT AND IMPLICIT MPC CONSTRAINTS

    # set implicit constraints/feasible sets for x and u at each time-step
    ocp.set_imp_constraints(stage_constraint_func=ocp_functions.stage_constraint_func,
                            term_constraint_func=ocp_functions.term_constraint_func)

    # set explicit state inequality constraints
    ocp.set_state_ineq_func(ocp_functions.state_ineq_constraint_func)

    ocp.set_constraints()

    ####################################################################################################################

    # INITIALIZE PD OPTIMIZER
    pdo = PDOptimizationMethod(ocp=ocp)

    ####################################################################################################################

    # INITIALIZE MPC
    mpc = MPC(ocp, pdo, opt_params, n_horizon, use_opt_method=use_opt_method)

    ####################################################################################################################
    # SIMULATE SYSTEM

    if simulate_system:
        simulator = Simulator(training_data_reader, ocp, mpc,
                              actual_next_state_func, ocp_functions.actual_stage_cost_func,
                              ocp_functions.actual_terminal_cost_func,
                              n_simulation_steps, next_state_meas_noises, numerical_bounds)

        init_state, actual_disturbances = ocp_functions.simulation_variables(n_states, n_disturbances,
                                                                             n_simulation_steps, n_horizon)

        next_state_sampling_threads = []
        if n_init_training_samples < n_training_samples:
            # LAUNCH THREADS TO FETCH SAMPLES OF DYNAMIC STATE AND COST FUNCTIONS IN PARALLEL WITH SIMULATION
            is_simulation_running = Event()
            for s, use_gp in enumerate(use_next_state_gp_approx):
                if use_gp:
                    next_state_sampling_threads.append(Thread(
                        target=next_state_gps[s].collect_training_data_thread_func,
                        args=(next_state_dfs[s], is_simulation_running, next_state_sampling_period,)))

            cost_sampling_thread = Thread(
                target=cost_gp.collect_training_data_thread_func, args=(cost_df, is_simulation_running,
                                                                        cost_sampling_period,)
            )

            simulation_thread = Thread(
                target=simulator.simulate_thread_func, args=(init_state, actual_disturbances, is_simulation_running,)
            )
            for thread in next_state_sampling_threads + [cost_sampling_thread, simulation_thread]:
                thread.start()
                thread.join()

        else:
            simulator.simulate(init_state, actual_disturbances)

        system_trajectory = [simulator.state_trajectory, simulator.input_trajectory, simulator.disturbance_trajectory,
                             simulator.cost_trajectory, simulator.regret_trajectory, simulator.ave_regret_trajectory,
                             simulator.ineq_constraint_trajectory, simulator.eq_constraint_trajectory]

        # if running an online algorithm
        if opt_params['maxiter'] < 50:
            # fetch equivalent offline simulation
            comp_sim_name = f'{"GP" if any(use_next_state_gp_approx + [use_cost_gp_approx]) else "Known Models"} ' \
                            f'Offline N={n_horizon}'
            comp_sim_dir = f'./results/{comp_sim_name}/trajectory.csv'
            comp_sim_df = pd.read_csv(comp_sim_dir)
            sim_df_cols = ['x0', 'u0', 'w0', 'cost', 'regret', 'average_regret', 'ineq_constraints', 'eq_constraints']
            comp_sim_df[sim_df_cols] = \
            comp_sim_df[sim_df_cols].applymap(
                lambda cell: np.asarray(re.findall('(-*\d*\.\d*e*-*\d*)', cell)).astype(float))
        else:
            comp_sim_df = None

        # export trajectories
        simulation_dict = {key: system_trajectory[sim_df_cols.index(key)].tolist() for key in sim_df_cols}
        simulation_df = pd.DataFrame(simulation_dict)
        simulation_df.to_csv(f'{sim_dir}/trajectory.csv', index_label='time_step')

        # export figures
        traj_fig, traj_ax = simulator.plot_trajectory(comp_sim_df)
        conv_fig, conv_ax, var_fig, var_ax = simulator.plot_convergence()
        anim_fig, anim_ax, anim = simulator.animate()
        traj_fig.savefig(f'{sim_dir}/traj_plot')
        traj_fig.savefig(f'{export_dir}/{simulation_name}_traj_plot')
        conv_fig.savefig(f'{sim_dir}/conv_plot')
        conv_fig.savefig(f'{export_dir}/{simulation_name}_conv_plot')
        var_fig.savefig(f'{sim_dir}/var_plot')
        var_fig.savefig(f'{export_dir}/{simulation_name}_var_plot')
        anim_fig.savefig(f'{sim_dir}/anim_plot')
        writer = animation.FFMpegWriter(fps=15)
        anim.save(f'{sim_dir}/pendulum.mp4', writer=writer)

        ####################################################################################################################

    # TEST DYNAMIC STATE AND COST FUNCTION GPS

    if any(use_next_state_gp_approx + [use_cost_gp_approx]):
        # generate current test input data
        current_state_test = np.hstack(
            [np.linspace(numerical_bounds[x][0], numerical_bounds[x][1],
                         n_test_samples)[np.newaxis, :].T for x in states])

        current_input_test = np.hstack(
            [np.linspace(numerical_bounds[u][0], numerical_bounds[u][1],
                         n_test_samples)[np.newaxis, :].T for u in inputs])

        if n_disturbances:
            current_disturbance_test = np.hstack([0 * np.linspace(numerical_bounds[w][0],
                                                                  numerical_bounds[w][1],
                                                                  n_test_samples)[np.newaxis, :].T
                                                  for w in disturbances])
        else:
            current_disturbance_test = np.zeros((n_test_samples, n_disturbances))

    if any(use_next_state_gp_approx):
        # PLOT DYNAMIC STATE GP APPROXIMATION

        input_dims = [(states + inputs + disturbances).index(l) for l in
                      next_state_input_labels]  # states.index(state_input_label)#system_model.x.keys().index(state_input_label)
        next_state_input_labels = [f'$z_{{k, {i}}}$' for i in input_dims]
        output_dims = [states.index(l) for l in
                       next_state_output_labels]  # states.index(state_output_label)#system_model.x.keys().index(state_output_label)
        next_state_output_labels = [f'$x_{{k + 1, {i}}}$' for i in output_dims]

        # generate lagged test input data
        lagged_output_test, lagged_input_test, lagged_disturbance_test = \
            training_data_reader.generate_lagged_data(n_test_samples, current_state_test,
                                                      current_input_test, current_disturbance_test)

        next_state_x_test = np.hstack([lagged_output_test, lagged_input_test, lagged_disturbance_test])

        if model_next_state_error:
            # get the true next state error
            next_state_true = np.vstack([actual_next_state_func(z=next_state_x_test[s])
                                         - next_state_prior_func(z=next_state_x_test[s])
                                         for s in range(n_test_samples)])
        else:
            # get the true next state
            next_state_true = np.vstack([actual_next_state_func(z=next_state_x_test[s])
                                         for s in range(n_test_samples)])

    for s, use_gp in enumerate(use_next_state_gp_approx):
        if use_gp:
            # y_pred, y_std = next_state_error_gp.gp_fit.predict(x_next_state_test)

            # plot gp approximation function values at the test inputs
            # plot actual function values at the test inputs

            # fetch the predicted error is using the next state error gp,
            # else the predicted next state if using the next state gp
            next_state_pred, next_state_std = next_state_gps[s].predict(next_state_x_test)

            next_state_figs, next_state_axes = next_state_gps[s].plot(next_state_x_test,
                                                                      next_state_true[:, s][:, np.newaxis],
                                                                      next_state_pred,
                                                                      next_state_std,
                                                                      input_dims=input_dims,
                                                                      input_labels=next_state_input_labels,
                                                                      output_dims=output_dims,
                                                                      output_labels=next_state_output_labels,
                                                                      title='Next State Error'
                                                                      if model_next_state_error
                                                                      else 'Next State')
            for fig in next_state_figs:
                fig.savefig(f'{sim_dir}/_next_state_error_{s}_gp' if model_next_state_error
                            else f'/{simulation_name}_next_state_{s}_gp')
                fig.savefig(f'{export_dir}/{simulation_name}_next_state_error_{s}_gp' if model_next_state_error
                        else f'/{simulation_name}_next_state_{s}_gp')

            next_state_df = next_state_gps[s].generate_results_df(next_state_true[:, s][:, np.newaxis],
                                                                        next_state_x_test)
            next_state_df.to_csv(sim_dir + f'/next_state_error_{s}_results' if model_next_state_error
                                       else f'/next_state_{s}_results')

            gp_results = {'Prediction Name': simulation_name,
                          'Function': f'Next State {s}',
                          'No. Training Samples': n_training_samples,
                          'Length Scale': next_state_length_scales[s],
                          'Output Variance': next_state_output_variances[s],
                          'Measurement Noise': next_state_meas_noises[s],
                          'Score': next_state_gps[s].scores[0]}

            existing_row_indices = (gp_results_df['Prediction Name'] == simulation_name) \
                                   & (gp_results_df['Function'] == gp_results['Function'])
            if existing_row_indices.any(axis=0):
                gp_results_df.loc[existing_row_indices] = gp_results
            else:
                gp_results_df = gp_results_df.append(gp_results, ignore_index=True)
                gp_results_df = gp_results_df.reindex()

            gp_results_df.to_csv('./results/gp_results.csv')

            score_fig, score_ax = next_state_gps[s].plot_score(
                gp_results_df.loc[gp_results_df['Function'] == gp_results['Function']])

            score_fig.savefig(f'{export_dir}/{simulation_name}_next_state_error_{s}_gp_score' if model_next_state_error
                        else f'/{simulation_name}_next_state_{s}_gp_score')

            next_state_table_dir = f'{export_dir}/next_state_gp_{s}_results.txt'

            prediction_number = len(open(next_state_table_dir).readlines()) + 1 if os.path.exists(next_state_table_dir) \
                else 1

            next_state_table_row = f'{prediction_number} & {n_training_samples} & {next_state_length_scales[s]} ' \
                                   f'& {next_state_output_variances[s]} ' \
                                   f'& {next_state_meas_noises[s]} & {next_state_gps[s].scores[0]}\\\\\n'

            next_state_table_file = open(f'{next_state_table_dir}', 'a')
            next_state_table_file.write(next_state_table_row)
            next_state_table_file.close()

            ################################################################################################################

        # PLOT COST GP APPROXIMATION

    if use_cost_gp_approx:
        cost_x_test = np.hstack([current_state_test, current_input_test, current_disturbance_test])
        # cost_pred, cost_std = cost_gp.gp_fit.predict(cost_x_test)

        input_dims = [(states + inputs).index(l) for l in
                      cost_input_labels]  # system_model.x.keys().index(state_input_label)
        output_dims = [0 for i in input_dims]
        cost_input_labels = [f'$z_{{k, {i}}}$' for i in input_dims]
        cost_output_labels = [f'$\overline{{l}}(z_k)$' for i in input_dims]

        cost_pred, cost_std = cost_gp.predict(cost_x_test)
        cost_true = np.vstack([[ocp_functions.actual_stage_cost_func(z_stage=cost_x_test[s])
                                for d in range(len(input_dims))] for s in range(n_test_samples)])

        # plot gp approximation function values at the test inputs
        cost_figs, cost_axes = cost_gp.plot(cost_x_test,
                                            cost_true,
                                            cost_pred,
                                            cost_std,
                                            input_dims=input_dims,
                                            input_labels=cost_input_labels,
                                            output_dims=output_dims,
                                            output_labels=cost_output_labels,
                                            title="Cost")

        cost_figs[0].savefig(f'{sim_dir}/cost_gp')
        cost_figs[0].savefig(f'{export_dir}/{simulation_name}_cost_gp')

        cost_df = cost_gp.generate_results_df(cost_true, cost_x_test)
        cost_df.to_csv(f'{sim_dir}/cost_results')

        gp_results = {'Prediction Name': simulation_name,
                      'Function': f'Stage Cost',
                      'No. Training Samples': n_training_samples,
                      'Length Scale': cost_length_scale,
                      'Output Variance': cost_output_variance,
                      'Measurement Noise': cost_meas_noise,
                      'Score': cost_gp.scores[0]}

        existing_row_indices = (gp_results_df['Prediction Name'] == simulation_name) \
                               & (gp_results_df['Function'] == gp_results['Function'])
        if existing_row_indices.any(axis=0):
            gp_results_df.loc[existing_row_indices] = gp_results
        else:
            gp_results_df = gp_results_df.append(gp_results, ignore_index=True)
            gp_results_df = gp_results_df.reindex()

        gp_results_df.to_csv('./results/gp_results.csv')

        score_fig, score_ax = cost_gp.plot_score(gp_results_df.loc[gp_results_df['Function'] == gp_results['Function']])

        score_fig.savefig(f'{export_dir}/{simulation_name}_cost_gp_score')

        cost_table_dir = f'{export_dir}/cost_gp_results.txt'
        prediction_number = len(open(cost_table_dir).readlines()) + 1 if os.path.exists(cost_table_dir) else 1
        cost_table_row = f'{prediction_number} & {n_training_samples} & {cost_length_scale} & {cost_output_variance} ' \
                         f'& {cost_meas_noise} & {cost_gp.scores[0]}\\'
        cost_table_file = open(cost_table_dir, 'a')
        cost_table_file.write(cost_table_row)
        cost_table_file.close()
    ####################################################################################################################

    # COMPARE 2 SIMULATIONS
    if compare_simulations and len(simulation_comparisons) == 2:
        sim_a_name = simulation_comparisons[0]
        sim_b_name = simulation_comparisons[1]
        df_a = pd.read_csv(sim_a_name + '/' + 'trajectory.csv', engine="python", index_col=0)
        df_b = pd.read_csv(sim_b_name + '/' + 'trajectory.csv', engine="python", index_col=0)
        simulation_comparator = SimulationComparator(df_a, df_b)
        comp_fig, comp_ax = simulation_comparator.plot_rel_error()

        comp_dir = './results/Compared Simulations/'

        comp_fig.savefig(comp_dir + 'traj_plot')

    return


if __name__ == '__main__':
    # define experiment and system input parameters, measurement vector consists of states and cost
    default_input_params = {'n_training_samples': 100, 'n_init_training_samples': 100, 'n_test_samples': 100,
                            'n_simulation_steps': 2000, 'n_horizon': 3, 't_step': .1,
                            'state_lag': 0, 'input_lag': 0, 'disturbance_lag': 0,
                            'next_state_length_scales': [30, 3.14], 'next_state_output_variances': [50, 20],  # 1.6, 50
                            'cost_length_scale': 20, 'cost_output_variance': 100.,  # 60, 1000; 12, 40

                            'next_state_meas_noises': [0.01, 0.1], 'cost_meas_noise': 1.0,
                            'next_state_sampling_period': 1, 'cost_sampling_period': 1,

                            'xtol': 1e-3, 'maxiter': 500, 'lambda_prior': 0, 'mu_prior': 0,
                            'alpha': .05, 'eta': 1., 'eps': 1.,

                            'reset_simulation_results': False,
                            'use_opt_method': True,
                            'use_cost_gp_approx': False,
                            'use_next_state_gp_approx': [False, True],
                            'model_next_state_error': True,
                            'model_type': 'discrete',
                            'model_nonlinear_next_state': True,
                            'simulate_system': True,
                            'compare_simulations': False,
                            'simulation_name': 'Known Models Offline N=3',
                            'simulation_comparisons': ['GP Offline N=5', 'GP Online N=5 maxiter=1'],
                            'cost_input_labels': ['theta'],
                            'next_state_input_labels': ['theta'],
                            'next_state_output_labels': ['theta_dot'],
                            'imp_bounds_only': True}

    if not os.path.exists('./results'):
        os.mkdir('./results')

    if not os.path.exists('./results/exports'):
        os.mkdir('./results/exports')

    if not os.path.exists('./results/Compared Simulations'):
        os.makedirs('./results/Compared Simulations')

    reset_simulation_results = default_input_params['reset_simulation_results']
    if reset_simulation_results:
        if os.path.exists('./results/simulation_parameters.csv'):
            os.remove('./results/simulation_parameters.csv')

        if os.path.exists('./results/gp_results.csv'):
            os.remove('./results/gp_results.csv')

        for root, _, files in os.walk('./exports'):
            for file in files:
                os.remove(os.path.join(f'{root}', file))

    simulations = [
        # 0:9 GP Simulations
        {'simulation_name': 'GP Ntr=5', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 5, 'n_training_samples': 5},
        {'simulation_name': 'GP Ntr=10', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 10, 'n_training_samples': 10},
        {'simulation_name': 'GP Ntr=25', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 25, 'n_training_samples': 25},
        {'simulation_name': 'GP Ntr=50', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 50, 'n_training_samples': 50},
        {'simulation_name': 'GP Ntr=100', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 100, 'n_training_samples': 100},
        {'simulation_name': 'GP Ntr=250', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 250, 'n_training_samples': 250},
        {'simulation_name': 'GP Ntr=500', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 500, 'n_training_samples': 500},
        {'simulation_name': 'GP Ntr=1000', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 1000, 'n_training_samples': 1000},
        {'simulation_name': 'GP Ntr=2000', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'simulate_system': False, 'n_init_training_samples': 2000, 'n_training_samples': 2000},

        # 9:12 Known Model Offline Simulations
        {'simulation_name': 'Known Models Offline N=3', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 2000, 'n_horizon': 3, 'eta': 1.0},
        {'simulation_name': 'Known Models Offline N=5', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 2000, 'n_horizon': 5, 'eta': 1.0},
        {'simulation_name': 'Known Models Offline N=8', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 2000, 'n_horizon': 8, 'eta': 1.0},

        # 12:15 GP Function Offline Simulations
        {'simulation_name': 'GP Offline N=3', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'maxiter': 2000, 'n_horizon': 3, 'eta': 0.2}, # drifting oscillations
        {'simulation_name': 'GP Offline N=5', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'maxiter': 2000, 'n_horizon': 5, 'eta': 0.4},
        {'simulation_name': 'GP Offline N=8', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'maxiter': 2000, 'n_horizon': 8, 'eta': 0.7},

        # 15:18 Known Models Online Simulations
        {'simulation_name': 'Known Models Online N=5 maxiter=1', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 1, 'n_horizon': 5, 'eta': 1.0},
        {'simulation_name': 'Known Models Online N=5 maxiter=5', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 5, 'n_horizon': 5, 'eta': 1.0},
        {'simulation_name': 'Known Models Online N=5 maxiter=10', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 10, 'n_horizon': 5, 'eta': 1.0},

        # 18:21 GP Function Online Simulations
        {'simulation_name': 'GP Online N=5 maxiter=1', 'use_cost_gp_approx': True,
         'use_next_state_gp_approx': [False, True],
         'maxiter': 1, 'n_horizon': 5, 'eta': 0.4},
        {'simulation_name': 'GP Online N=5 maxiter=5', 'use_cost_gp_approx': True,
         'use_next_state_gp_approx': [False, True],
         'maxiter': 5, 'n_horizon': 5, 'eta': 0.4},
        {'simulation_name': 'GP Online N=5 maxiter=10', 'use_cost_gp_approx': True,
         'use_next_state_gp_approx': [False, True],
         'maxiter': 10, 'n_horizon': 5, 'eta': 0.4}
    ]

    for custom_input_params in simulations[9:]:
        new_input_params = {key: custom_input_params[key] if key in custom_input_params else default_input_params[key]
                            for key in default_input_params}
        main(new_input_params)
