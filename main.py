# TODO 1) check that next_state_prior_func is right in read_training_data and in model method
#  2) test different length scales, 3) tune params until GP next state works 4) run simulation in parallel
#  to given data

import numpy as np
# from InputReader import InputReader
from matplotlib import animation
from GP import StageCostGP, NextStateGP
from OCP import OCP
from MPC import MPC
from Simulator import Simulator, SimulationComparator
from DataReader import DataReader
from Models import VirtualPowerPlant
from threading import Thread, Event
import pandas as pd
from PDOptimizationMethod import PDOptimizationMethod
import os
from helper_functions import gradient_func, jacobian_func
import re
from GenerateExports import GenerateExports


def main(input_params):
    ####################################################################################################################

    # READ INPUT PARAMETERS

    # NOTE: if using GP approx, set model_type to discrete and return cont_func from true_next_state

    global next_state_error_train
    state_lag = input_params['state_lag']
    input_lag = input_params['input_lag']
    disturbance_lag = input_params['disturbance_lag']
    output_lag = input_params['output_lag']
    n_horizon = input_params['n_horizon']
    mpc_t_step = input_params['mpc_t_step']
    n_training_samples = input_params['n_training_samples']
    n_init_training_samples = input_params['n_init_training_samples']
    n_test_samples = input_params['n_test_samples']
    n_simulation_steps = input_params['n_simulation_steps']

    gp_sampling_periods = []

    opt_params = {key: input_params[key] for key in ['xtol', 'maxiter', 'lambda_prior', 'mu_prior',
                                                     'alpha', 'eta', 'eps']}

    # plot_gp = input_params['plot_gp']
    use_opt_method = input_params['use_opt_method']
    unknown_next_state_functions = input_params['unknown_next_state_functions']
    unknown_stage_cost_functions = input_params['unknown_stage_cost_functions']
    n_devices = input_params['n_devices']

    model_type = input_params['model_type']
    model_nonlinear_next_state = input_params['model_nonlinear_next_state']
    run_mpc = input_params['run_mpc']
    compare_simulations = input_params['compare_simulations']
    simulation_name = input_params['simulation_name']
    simulation_comparisons = input_params['simulation_comparisons']
    imp_bounds_only = input_params['imp_bounds_only']
    use_linear_test_values = input_params['use_linear_test_values']

    n_batteries = input_params['n_batteries']
    n_tcls = input_params['n_tcls']

    simulation_dir = input_params['simulation_dir']
    results_dir = f'./{simulation_dir}/results/{simulation_name}'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    else:
        for _, _, files in os.walk(results_dir):
            for file in files:
                os.remove(f'{results_dir}/{file}')

    # input_params = InputReader(input_params)
    ####################################################################################################################

    # IMPORT SPECIFIC OPTIMAL CONTROL PROBLEM FUNCTIONS
    model = VirtualPowerPlant(t_step=mpc_t_step, state_lag=state_lag, input_lag=input_lag,
                              disturbance_lag=disturbance_lag, output_lag=output_lag, n_horizon=n_horizon,
                              n_simulation_steps=n_simulation_steps, n_batteries=n_batteries, n_tcls=n_tcls,
                              n_inverted_pendulums=n_inverted_pendulums)

    ####################################################################################################################

    # INITIALISE OPTIMAL CONTROL PROBLEM
    ocp = OCP(n_horizon=n_horizon, state_lag=state_lag, input_lag=input_lag, disturbance_lag=disturbance_lag,
              output_lag=output_lag, mpc_t_step=mpc_t_step, model=model, imp_bounds_only=imp_bounds_only,
              model_type=model_type)

    ####################################################################################################################

    # INITIALISE DYNAMIC-STATE (ERROR) & COST GP APPROXIMATIONS

    # define dynamic state function GP approximation
    stage_cost_gps = []
    next_state_gps = []
    next_state_funcs = []
    next_state_jacobians = []
    stage_cost_funcs = []
    stage_cost_jacobians = []
    for d, func in enumerate(unknown_stage_cost_functions):
        if func['use_gp']:

            # define cost function GP approximation
            cost_gp = StageCostGP(func['output_variance'],
                                  func['meas_noise'],
                                  func['length_scale'],
                                  n_training_samples, n_test_samples, model.devices[d].n_states,
                                  model.devices[d].n_inputs, model.devices[d].n_disturbances)
            stage_cost_gps.append(cost_gp)

            # if modeling the difference between the observed function and the prior

            def stage_cost_func(z_stage, gp=cost_gp, dd=d):
                # first 0 index fetches the mean, second 0 index fetches the first row of the return predictions
                return model.devices[dd].stage_cost_prior_func(z_stage) + gp.predict(z_stage[np.newaxis, :])[0][0]

            def stage_cost_jacobian(z_stage, gp=cost_gp, dd=d):
                return model.devices[dd].stage_cost_prior_jacobian(z_stage) + gp.mean_jacobian(z_stage)
                # return np.vstack([gradient_func(model.devices[dd].true_stage_cost_func, z_stage)]) \
                #        + gp.mean_jacobian(z_stage)

        else:
            stage_cost_gps.append(None)

            stage_cost_func = model.true_stage_cost_func
            stage_cost_jacobian = model.true_stage_cost_jacobian

        stage_cost_funcs.append(stage_cost_func)
        stage_cost_jacobians.append(stage_cost_jacobian)

    def stage_cost_func(z_stage, k=0):
        # first 0 index fetches the mean, second 0 index fetches the first row of the return predictions
        model.set_simulation_step(k)
        stage_cost = model.system_stage_cost_func(z_stage)
        for d, dev_func in enumerate(stage_cost_funcs):
            stage_cost += dev_func(z_stage[model.device_zstage_indices[d]])
        return stage_cost

    def stage_cost_jacobian(z_stage, k=0):
        model.set_simulation_step(k)
        jac = model.system_stage_cost_jacobian(z_stage)
        for d, dev_jac in enumerate(stage_cost_jacobians):
            jac[0, model.device_zstage_indices[d]] = jac[0, model.device_zstage_indices[d]] \
                                                     + dev_jac(z_stage[model.device_zstage_indices[d]])
        return jac

    def terminal_cost_func(x_term, k=0):
        model.set_simulation_step(k)
        return model.terminal_cost_func(x_term)

    def terminal_cost_jacobian(x_term, k=0):
        model.set_simulation_step(k)
        return model.terminal_cost_jacobian(x_term)

    for d, funcs in enumerate(unknown_next_state_functions):
        for func in funcs:
            if func['use_gp']:
                next_state_gp = NextStateGP(func['output_variance'],
                                            func['meas_noise'],
                                            func['length_scale'],
                                            n_training_samples, n_test_samples,
                                            state_lag, input_lag, disturbance_lag, output_lag,
                                            model.devices[d].n_states, model.devices[d].n_inputs,
                                            model.devices[d].n_disturbances, model.devices[d].n_outputs)
                next_state_gps.append(next_state_gp)

                # modeling the difference between the observed function and the prior

                # call conditional probability calculator each time the function is needs
                def next_state_func(z_lagged, gp=next_state_gp, dd=d):
                    return model.devices[dd].next_state_prior_func(z_lagged) + gp.predict(z_lagged[np.newaxis, :])[0][0]

                def next_state_jacobian(z_lagged, gp=next_state_gp, dd=d):
                    # z_lagged = [x_lagged, u_lagged, w_lagged]
                    return model.devices[dd].next_state_prior_jacobian(z_lagged) + gp.mean_jacobian(z_lagged)
                    # return jacobian_func(model.devices[dd].next_state_prior_func, z_lagged) + gp.mean_jacobian(z_lagged)

            else:
                next_state_gps.append(None)

                # bind d variable as a default parameter to function so that it doesn't always refer to last possible state
                def next_state_func(z_lagged, dd=d):
                    return model.devices[dd].true_next_state_func(z_lagged)

                def next_state_jacobian(z_lagged, dd=d):
                    return jacobian_func(model.devices[dd].true_next_state_func, z_lagged)

            next_state_funcs.append(next_state_func)
            next_state_jacobians.append(next_state_jacobian)

    ####################################################################################################################

    # INITIALISE DYNAMIC-STATE AND COST FUNCTIONS, EXPLICIT AND IMPLICIT MPC CONSTRAINTS IN OCP

    # if using gp approximation
    # set the stage cost to the gp approximation call and the terminal cost the given function
    ocp.set_cost_funcs(stage_cost_func, terminal_cost_func)
    ocp.set_cost_jacobians(stage_cost_jacobian, terminal_cost_jacobian)
    ocp.set_next_state_jacobians(next_state_jacobians)
    ocp.set_stage_ineq_jacobian(model.stage_ineq_constraint_jacobian)

    # if using the gp approximation
    # set the discrete next state to the gp approximation call
    # else if using the true identified dynamic state and cost functions
    # set the discrete next state to the true next state function
    ocp.set_next_state_funcs(next_state_funcs)

    # set implicit constraints/feasible sets for x and u at each time-step
    ocp.set_imp_constraints(stage_constraint_func=model.stage_constraint_func,
                            term_constraint_func=model.term_constraint_func)

    # set explicit state inequality constraints
    ocp.set_stage_ineq_func(model.stage_ineq_constraint_func)

    ocp.set_constraints()

    ####################################################################################################################

    # STORE TRAINING DATA

    # instantiate a TrainingDataReader object to generate system measurements
    data_reader = DataReader(input_params, model.n_states, model.n_inputs, model.n_disturbances,
                             model.n_outputs, n_simulation_steps)
    gp_df_cols = ['x_train', 'y_train']
    stage_cost_training_data_dfs = []
    next_state_training_data_dfs = []
    for d, func in enumerate(unknown_stage_cost_functions):
        if func['use_gp']:

            if func['synthetic_data']:
                # get input and output synthesized training data
                x_train, x_train_ind, y_true, y_train, y_train_ind \
                    = data_reader.generate_training_data(model.devices[d], func, mpc_t_step,
                                                         model.devices[d].true_stage_cost_func,
                                                         model.devices[d].stage_cost_prior_func, is_state_gp=False,
                                                         output_dim=0)
                independent_data = func['training_data_path'] is None
                if independent_data:
                    x_train = x_train_ind
                    y_train = y_train_ind

            else:

                # get input and output true training data
                x_train, y_train = data_reader.read_training_data(func)

            # add cost training data to database
            cost_df = pd.DataFrame(columns=gp_df_cols)
            cost_df = data_reader.add_training_data(cost_df, x_train, y_train)

            stage_cost_training_data_dfs.append(cost_df)
        else:
            stage_cost_training_data_dfs.append(None)

    # for each state of each dives
    for d, funcs in enumerate(unknown_next_state_functions):
        for dim, func in enumerate(funcs):
            # if using a gp to estimate this state
            if func['use_gp']:
                # if there is true training data

                if func['synthetic_data']:
                    # get input and output synthesized training data
                    x_train, x_train_ind, y_true, y_train, y_train_ind \
                        = data_reader.generate_training_data(
                        model, func, mpc_t_step, model.devices[d].true_next_state_func,
                        model.devices[d].next_state_prior_func, is_state_gp=True, output_dim=dim)

                    independent_data = func['training_data_path'] is None
                    if independent_data:
                        x_train = x_train_ind
                        y_train = y_train_ind

                else:
                    # get input and output true training data Todo yprior is wrong
                    x_train, y_train = data_reader.read_training_data(func)

                # add to database
                next_state_df = pd.DataFrame(columns=gp_df_cols)
                next_state_df = data_reader.add_training_data(next_state_df, x_train, y_train)

                # add next state (error) training data to database
                next_state_training_data_dfs.append(next_state_df)

            else:
                next_state_training_data_dfs.append(None)

    ####################################################################################################################

    # FETCH INITIAL SAMPLES OF DYNAMIC STATE AND COST FUNCTIONS

    # collect function samples from DataFrame and add to GP training set
    # train gps with independent data (ie only a single feature nonzero for each sample) if using synthetic data
    for f, func in enumerate(unknown_stage_cost_functions):
        if func['use_gp']:
            stage_cost_gps[f].collect_training_data(stage_cost_training_data_dfs[f], n_init_training_samples)

    f = -1
    for d, device_funcs in enumerate(unknown_next_state_functions):
        for func in device_funcs:
            f += 1
            if func['use_gp']:
                next_state_gps[f].collect_training_data(next_state_training_data_dfs[f], n_init_training_samples)

    ####################################################################################################################

    # INITIALIZE PD OPTIMIZER
    pdo = PDOptimizationMethod(ocp=ocp)

    ####################################################################################################################

    # INITIALIZE MPC
    mpc = MPC(ocp, pdo, opt_params, n_horizon, use_opt_method=use_opt_method)

    ####################################################################################################################

    # TEST DYNAMIC STATE AND COST FUNCTION GPS
    for f, func in enumerate(unknown_stage_cost_functions):
        stage_cost_gps[f].test(data_reader, model.devices[f], func, n_test_samples, n_simulation_steps,
                               n_training_samples, mpc_t_step, results_dir, simulation_dir, simulation_name, run_mpc,
                               use_linear_test_values, f)

    f = -1
    for d, device_funcs in enumerate(unknown_next_state_functions):
        for func in device_funcs:
            f += 1
            next_state_gps[f].test(data_reader, model.devices[d], func, n_test_samples, n_simulation_steps,
                                   n_training_samples, mpc_t_step, results_dir, simulation_dir, simulation_name, run_mpc,
                                   use_linear_test_values, d, f)

    ####################################################################################################################
    # SIMULATE SYSTEM

    if run_mpc:

        mpc_inputs = {key: [input_params[key]]
                      for key in ['simulation_name', 'maxiter', 'n_training_samples', 'alpha',
                                  'eta', 'eps', 'n_horizon', 'n_simulation_steps']}
        if not os.path.exists(f'./{simulation_dir}/results/mpc_results.csv'):
            mpc_results_df = pd.DataFrame(mpc_inputs)
        else:
            mpc_results_df = pd.read_csv(f'./{simulation_dir}/results/mpc_results.csv', engine='python', index_col=0)
            mpc_results_df.drop(mpc_results_df.loc[
                                    mpc_results_df['simulation_name'] == simulation_name].index, inplace=True)

            existing_row_indices = (mpc_results_df['simulation_name'] == simulation_name)
            if existing_row_indices.any(axis=0):
                mpc_results_df.loc[existing_row_indices] = mpc_inputs
            else:
                mpc_results_df = mpc_results_df.append(mpc_inputs, ignore_index=True)
                mpc_results_df = mpc_results_df.reindex()

        mpc_results_df.to_csv(f'./{simulation_dir}/results/mpc_results.csv')

        simulator = Simulator(ocp, mpc, model, n_simulation_steps)

        true_states, true_disturbances = data_reader.read_true_data(n_horizon, unknown_next_state_functions, mpc_t_step)

        stage_cost_gp_sampling_threads = []
        next_state_gp_sampling_threads = []
        if n_init_training_samples < n_training_samples:
            # LAUNCH THREADS TO FETCH SAMPLES OF DYNAMIC STATE AND COST FUNCTIONS IN PARALLEL WITH SIMULATION
            is_simulation_running = Event()

            for f, func in enumerate(unknown_stage_cost_functions):
                if func['use_gp']:
                    stage_cost_gp_sampling_threads.append(Thread(
                        target=stage_cost_gps[f].collect_training_data_thread_func,
                        args=(stage_cost_training_data_dfs[f], is_simulation_running, func['sampling_t_step'],)))

            f = -1
            for d, device_funcs in enumerate(unknown_next_state_functions):
                for func in device_funcs:
                    f += 1
                    if func['use_gp']:
                        next_state_gp_sampling_threads.append(Thread(
                            target=next_state_gps[f].collect_training_data_thread_func,
                            args=(next_state_training_data_dfs[f], is_simulation_running, func['sampling_t_step'],)))

            simulation_thread = Thread(
                target=simulator.simulate_thread_func,
                args=(model.init_state, true_states, true_disturbances, is_simulation_running,)
            )

            threads = stage_cost_gp_sampling_threads + next_state_gp_sampling_threads + [simulation_thread]
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        else:
            simulator.simulate(model.init_state, true_states, true_disturbances)

        system_trajectory = [simulator.state_trajectory, simulator.input_trajectory, simulator.disturbance_trajectory,
                             simulator.cost_trajectory, simulator.regret_trajectory, simulator.ave_regret_trajectory,
                             simulator.ineq_constraint_trajectory, simulator.eq_constraint_trajectory]

        sim_df_cols = ['x0', 'u0', 'w0', 'cost', 'regret', 'average_regret', 'ineq_constraints', 'eq_constraints']
        # if running an online algorithm
        if opt_params['maxiter'] <= 50:
            # fetch equivalent offline simulation
            use_gp = any([func["use_gp"] for func in unknown_stage_cost_functions]) \
                     or any([func['use_gp'] for func in device_funcs for device_funcs in unknown_next_state_functions])
            comp_sim_name = f'{"GP" if use_gp else "Known Models"} Offline N={n_horizon}'
            comp_results_dir = f'./{simulation_dir}/results/{comp_sim_name}/trajectory.csv'
            comp_sim_df = pd.read_csv(comp_results_dir)
            comp_sim_df[sim_df_cols] = comp_sim_df[sim_df_cols].applymap(
                lambda cell: np.asarray(re.findall('(-*\d*\.\d*e*-*\d*)', cell)).astype(float))
            if len(comp_sim_df.index) != n_simulation_steps:
                comp_sim_df = None
        else:
            comp_sim_df = None

        # export trajectories
        simulation_dict = {key: system_trajectory[sim_df_cols.index(key)].tolist() for key in sim_df_cols}
        simulation_df = pd.DataFrame(simulation_dict)
        simulation_df.to_csv(f'{results_dir}/trajectory.csv', index_label='time_step')

        # export figures
        traj_fig, traj_ax = simulator.plot_trajectory(comp_sim_df)
        conv_fig, conv_ax, var_fig, var_ax = simulator.plot_convergence()
        traj_fig.savefig(f'{results_dir}/traj_plot')
        conv_fig.savefig(f'{results_dir}/conv_plot')
        var_fig.savefig(f'{results_dir}/var_plot')

        if n_inverted_pendulums == 1:
            writer = animation.FFMpegWriter(fps=15)
            anim_fig, anim_ax, anim = simulator.animate()
            anim_fig.savefig(f'{results_dir}/anim_plot')
            anim.save(f'{results_dir}/pendulum.mp4', writer=writer)

        ####################################################################################################################

    # COMPARE 2 SIMULATIONS
    if compare_simulations and len(simulation_comparisons) == 2:
        sim_a_name = simulation_comparisons[0]
        sim_b_name = simulation_comparisons[1]
        df_a = pd.read_csv(sim_a_name + '/' + 'trajectory.csv', engine="python", index_col=0)
        df_b = pd.read_csv(sim_b_name + '/' + 'trajectory.csv', engine="python", index_col=0)
        simulation_comparator = SimulationComparator(df_a, df_b)
        comp_fig, comp_ax = simulation_comparator.plot_rel_error()

        comp_dir = f'./{simulation_dir}/results/Compared Simulations/'

        comp_fig.savefig(comp_dir + 'traj_plot')

    return


if __name__ == '__main__':

    # define experiment and system input parameters, measurement vector consists of states and cost
    n_tcls = 1
    n_batteries = 0
    n_inverted_pendulums = 0
    default_input_params = {'n_training_samples': 100, 'n_init_training_samples': 100, 'n_test_samples': 100,
                            'n_simulation_steps': 200, 'n_horizon': 1, 'mpc_t_step': 1.,
                            'state_lag': 0, 'input_lag': 0, 'disturbance_lag': 0, 'output_lag': 0,

                            # 'next_state_length_scales': [50, np.pi], 'next_state_output_variances': [50, 15],  # 1.6, 50
                            # 'cost_length_scale': 2 * np.pi, 'cost_output_variance': 1000.,  # 60, 1000; 12, 40

                            # 'next_state_meas_noises': [0.01, 0.01], 'cost_meas_noise': 0.01,
                            # 'next_state_sampling_period': 1, 'cost_sampling_period': 1,

                            'xtol': 1e-3, 'maxiter': 500, 'lambda_prior': 0, 'mu_prior': 0,
                            'alpha': .05, 'eta': 1., 'eps': 1.,

                            'simulation_dir': 'network',
                            'n_batteries': n_batteries,
                            'n_tcls': n_tcls,
                            'n_inverted_pendulums': n_inverted_pendulums,
                            'n_devices': n_tcls + n_batteries + n_inverted_pendulums,
                            'use_opt_method': True,
                            'unknown_stage_cost_functions':
                                [{'use_gp': True,
                                  'training_data_path': f'./device_data/battery_{d + 1}_cost_data.csv',
                                  'state_cols': [f'soc_{d + 1}'], 'input_cols': [f'P_{d + 1}'],
                                  'disturbance_cols': [f'P_ref_{d + 1}'],
                                  'sampling_t_step': 10 * 60,
                                  'length_scale': [16000, 16000, 16000],
                                  'output_variance': 2 * 1e8,
                                  'meas_noise': 0.01,
                                  'model_error': True,
                                  'function_type': 'stage_cost',
                                  'synthetic_data': True,
                                  'input_labels': [f'P_{d + 1}']}
                                 for d in range(n_batteries)] + \
                                [{'use_gp': True,
                                  'training_data_path': f'./device_data/tcl_{d + 1}_cost_data.csv',
                                  # temperature outside building, temperature, solar irradiance,
                                  # internal irradiance, mass air flow, discharge air temperature
                                  'state_cols': [f'T_zone_{d + 1}'], 'input_cols': [f'ms_dot_{d + 1}', f'T_da_{d + 1}'],
                                  'disturbance_cols': ['T_outside', f'Q_solar_{d + 1}', f'Q_internal_{d + 1}'],
                                  'sampling_t_step': 10 * 60,
                                  'length_scale': [8, 8, 8, 8, 8, 8],
                                  'output_variance': 10,
                                  'meas_noise': 0.01,
                                  'model_error': True,
                                  'function_type': 'stage_cost',
                                  'synthetic_data': True,
                                  'input_labels': [f'T_zone_{d + 1}']}
                                 for d in range(n_tcls)] + \
                                [{'use_gp': True,
                                  'training_data_path': None,
                                  'state_cols': [f'theta_{d + 1}', f'theta_dot_{d + 1}'],
                                  'input_cols': [f'u_{d + 1}'],
                                  'disturbance_cols': [], 'output_state_col': [f'theta_dot_{d + 1}'],
                                  'sampling_t_step': 1,
                                  'length_scale': [2 * np.pi, 1e-6, 1e-6],
                                  'output_variance': 1000,
                                  'meas_noise': 0.01,
                                  'model_error': True,
                                  'function_type': 'stage_cost',
                                  'synthetic_data': True,
                                  'input_labels': [f'theta_{d + 1}', f'theta_dot_{d + 1}']}
                                 for d in range(n_inverted_pendulums)],

                            'unknown_next_state_functions':
                                [[{'use_gp': True,
                                   'training_data_path': f'./device_data/battery_{d + 1}_data.csv',
                                   'state_cols': [f'soc_{d + 1}'], 'input_cols': [f'P_{d + 1}'],
                                   'disturbance_cols': [f'P_ref_{d + 1}'],
                                   'output_state_col': [f'soc_{d + 1}'],
                                   'sampling_t_step': 1,
                                   'length_scale': [0.001, 10, 0.001],
                                   'output_variance': 0.55,
                                   'meas_noise': 0.05,
                                   'model_error': True,
                                   'function_type': 'state',
                                   'synthetic_data': False,
                                   'input_labels': [f'soc_{d + 1}', f'P_{d + 1}'],
                                   'output_labels': [f'soc_{d + 1}']}] for d in range(n_batteries)] + \
                                [[{'use_gp': True,
                                   'training_data_path': f'./device_data/tcl_{d + 1}_data.csv',
                                   'state_cols': [f'T_zone_{d + 1}'], 'input_cols': [f'ms_dot_{d + 1}', 'T_da'],
                                   'disturbance_cols': ['T_outside', f'Q_solar_{d + 1}', f'Q_internal_{d + 1}'],
                                   'output_state_col': [f'T_zone_{d + 1}'],
                                   'sampling_t_step': 5 * 60,
                                   'length_scale': [1., 0.05, 0.5, 1, 1, 1],
                                   'output_variance': 0.7,
                                   'meas_noise': 0.01,
                                   'model_error': True,
                                   'function_type': 'state',
                                   'synthetic_data': False,
                                   'input_labels': [f'T_zone_{d + 1}', f'm_dot_a_{d + 1}', 'T_a'],
                                   'output_labels': [f'T_zone_{d + 1}']}] for d in range(n_tcls)] + \
                                [[{'use_gp': False},
                                  {'use_gp': True,
                                   'training_data_path': None,
                                   'state_cols': [f'theta_{d + 1}', f'theta_dot_{d + 1}'], 'input_cols': [f'u_{d + 1}'],
                                   'disturbance_cols': [],
                                   'sampling_t_step': 1,
                                   'length_scale': [np.pi, 10, 0.1],
                                   'output_variance': 15,
                                   'meas_noise': 0.01,
                                   'model_error': True,
                                   'function_type': 'stage_cost',
                                   'synthetic_data': True,
                                   'input_labels': [f'theta_{d + 1}', f'theta_dot_{d + 1}']}]
                                 for d in range(n_inverted_pendulums)],
                            'model_type': 'discrete',
                            'model_nonlinear_next_state': True,
                            'run_mpc': True,
                            'compare_simulations': False,
                            'simulation_name': 'Known Models Offline N=3',
                            'simulation_comparisons': ['GP Offline N=5', 'GP Online N=5 maxiter=1'],

                            # 'cost_input_labels': ['theta'],
                            # 'next_state_input_labels': ['theta', 'theta_dot', 'u1'],
                            # 'next_state_output_labels': ['theta_dot'],

                            'imp_bounds_only': True,
                            'use_linear_test_values': True}

    simulation_dir = default_input_params['simulation_dir']
    if not os.path.exists(f'./{simulation_dir}'):
        os.mkdir(f'./{simulation_dir}')

    if not os.path.exists(f'./{simulation_dir}/results'):
        os.mkdir(f'./{simulation_dir}/results')

    if not os.path.exists(f'./{simulation_dir}/training_data'):
        os.mkdir(f'./{simulation_dir}/training_data')

    if not os.path.exists(f'./{simulation_dir}/results/Compared Simulations'):
        os.makedirs(f'./{simulation_dir}/results/Compared Simulations')

    simulations = [
        # 0:9 GP Simulations
        {'simulation_name': 'GP Ntr=5', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 5, 'n_training_samples': 5},
        {'simulation_name': 'GP Ntr=10', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 10, 'n_training_samples': 10},
        {'simulation_name': 'GP Ntr=25', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 25, 'n_training_samples': 25},
        {'simulation_name': 'GP Ntr=50', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 50, 'n_training_samples': 50},
        {'simulation_name': 'GP Ntr=100', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 100, 'n_training_samples': 100},
        {'simulation_name': 'GP Ntr=250', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 250, 'n_training_samples': 250},
        {'simulation_name': 'GP Ntr=500', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 500, 'n_training_samples': 500},
        {'simulation_name': 'GP Ntr=1000', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 1000, 'n_training_samples': 1000},
        {'simulation_name': 'GP Ntr=2000', 'use_cost_gp_approx': True, 'use_next_state_gp_approx': [False, True],
         'run_mpc': False, 'n_init_training_samples': 2000, 'n_training_samples': 2000},

        # 9:12 Known Model Offline Simulations
        {'simulation_name': 'Known Models Offline N=3', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 2000, 'n_horizon': 3, 'eta': 1.},  # good
        {'simulation_name': 'Known Models Offline N=5', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 2000, 'n_horizon': 5, 'eta': 1.},  # good
        {'simulation_name': 'Known Models Offline N=8', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 2000, 'n_horizon': 8, 'eta': 1.},  # good

        # 12:16 Known Models Online Simulations
        {'simulation_name': 'Known Models Online N=3 maxiter=50', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 50, 'n_horizon': 3, 'eta': 1.},  # good
        {'simulation_name': 'Known Models Online N=3 maxiter=25', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 25, 'n_horizon': 3, 'eta': 1.},  # good
        {'simulation_name': 'Known Models Online N=3 maxiter=10', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 10, 'n_horizon': 3, 'eta': 1.},  # good
        {'simulation_name': 'Known Models Online N=3 maxiter=5', 'use_cost_gp_approx': False,
         'use_next_state_gp_approx': [False, False],
         'maxiter': 5, 'n_horizon': 3, 'eta': 1.},  # good

        # 16:19 GP Function Offline Simulations
        {'simulation_name': 'GP Offline N=3', 'maxiter': 2000, 'n_horizon': 1, 'eta': .5, 'alpha': 0.01},
        {'simulation_name': 'GP Offline N=5', 'maxiter': 2000, 'n_horizon': 5, 'eta': 1.05},  # good
        {'simulation_name': 'GP Offline N=8', 'maxiter': 2000, 'n_horizon': 8, 'eta': 1.05},

        # 20:25 GP Function Online Simulations
        # {'simulation_name': 'GP Online N=3 maxiter=1', 'use_cost_gp_approx': True,
        #  'use_next_state_gp_approx': [False, True],
        #  'maxiter': 1, 'n_horizon': 3, 'eta': .8},#needs smaller step-size
        # {'simulation_name': 'GP Online N=3 maxiter=5', 'use_cost_gp_approx': True,
        #  'use_next_state_gp_approx': [False, True],
        #  'maxiter': 5, 'n_horizon': 3, 'eta': .8},#needs greater-step-szie
        # {'simulation_name': 'GP Online N=3 maxiter=10', 'use_cost_gp_approx': True,
        #  'use_next_state_gp_approx': [False, True],
        #  'maxiter': 10, 'n_horizon': 3, 'eta': .8},
        {'simulation_name': 'GP Online N=3 maxiter=25', 'use_cost_gp_approx': True,
         'use_next_state_gp_approx': [False, True],
         'maxiter': 25, 'n_horizon': 3, 'eta': 1.05},
        {'simulation_name': 'GP Online N=3 maxiter=50', 'use_cost_gp_approx': True,
         'use_next_state_gp_approx': [False, True],
         'maxiter': 50, 'n_horizon': 3, 'eta': 1.05}

    ]

    for custom_input_params in simulations[4:5] + simulations[16:17]:
        new_input_params = {key: custom_input_params[key] if key in custom_input_params else default_input_params[key]
                            for key in default_input_params}
        main(new_input_params)

    generate_exports = GenerateExports()
    generate_exports.generate_gp_params_table()
    generate_exports.generate_mpc_params_table()
    generate_exports.copy_figures()
