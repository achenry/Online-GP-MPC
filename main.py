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
import shutil
from simulation_params import default_input_params, simulations, num_devices, stage_cost_funcs, next_state_funcs


def main(input_params):
    ####################################################################################################################

    # READ INPUT PARAMETERS

    # NOTE: if using GP approx, set model_type to discrete and return cont_func from true_next_state

    state_lag = input_params['state_lag']
    input_lag = input_params['input_lag']
    disturbance_lag = input_params['disturbance_lag']
    output_lag = input_params['output_lag']
    n_horizon = input_params['n_horizon']
    mpc_t_step = input_params['mpc_t_step']
    stage_cost_max_n_training_samples = input_params['stage_cost_max_n_training_samples']
    next_state_max_n_training_samples = input_params['next_state_max_n_training_samples']
    stage_cost_n_init_training_samples = input_params['stage_cost_n_init_training_samples']
    next_state_n_init_training_samples = input_params['next_state_n_init_training_samples']
    # n_test_samples = min(input_params['n_test_samples'], int(next_state_max_n_training_samples - 1))
    n_test_samples = input_params['n_test_samples']
    # input_params['n_test_samples'] = n_test_samples
    n_simulation_steps = input_params['n_simulation_steps']
    test_offset = input_params['test_offset']
    use_gp = input_params['use_gp']
    trajectory_bounds = input_params['trajectory_bounds']

    opt_params = {key: input_params[key] for key in ['xtol', 'maxiter', 'init_maxiter', 'lambda_prior', 'mu_prior',
                                                     'alpha', 'eta_ineq', 'eta_eq', 'eps']}

    use_opt_method = input_params['use_opt_method']

    model_type = input_params['model_type']
    run_mpc = input_params['run_mpc']
    run_state_gp = input_params['run_state_gp']
    run_cost_gp = input_params['run_cost_gp']
    compare_simulations = input_params['compare_simulations']
    simulation_name = input_params['simulation_name']
    simulation_comparisons = input_params['simulation_comparisons']
    imp_bounds_only = input_params['imp_bounds_only']
    use_linear_test_values = input_params['use_linear_test_values']

    n_inverted_pendulums = num_devices['ip']
    n_batteries = num_devices['batt']
    n_tcls = num_devices['tcl']

    stage_cost_funcs = input_params['stage_cost_funcs']
    next_state_funcs = input_params['next_state_funcs']

    unknown_stage_cost_functions = [stage_cost_funcs['batt'][d]
                                    for d in range(num_devices['batt'])] + \
                                   [stage_cost_funcs['tcl'][d]
                                    for d in range(num_devices['tcl'])] + \
                                   [stage_cost_funcs['ip'][d]
                                    for d in range(num_devices['ip'])]

    unknown_next_state_functions = [next_state_funcs['batt'][d]
                                    for d in range(num_devices['batt'])] + \
                                   [next_state_funcs['tcl'][d]
                                    for d in range(num_devices['tcl'])] + \
                                   [next_state_funcs['ip'][d]
                                    for d in range(num_devices['ip'])]

    gp_std_threshold = input_params['gp_std_threshold']
    zero_prior = input_params['zero_prior']
    # input_params = InputReader(input_params)

    ####################################################################################################################

    # IMPORT SPECIFIC OPTIMAL CONTROL PROBLEM FUNCTIONS
    model = VirtualPowerPlant(system_cost_weight=input_params['R'], t_step=mpc_t_step, state_lag=state_lag,
                              input_lag=input_lag, disturbance_lag=disturbance_lag, output_lag=output_lag,
                              n_horizon=n_horizon, n_simulation_steps=n_simulation_steps,
                              n_batteries=n_batteries, n_tcls=n_tcls, n_inverted_pendulums=n_inverted_pendulums)

    ####################################################################################################################

    # INITIALISE DYNAMIC-STATE (ERROR) & COST GP APPROXIMATIONS

    # define dynamic state function GP approximation
    stage_cost_gps = []
    next_state_gps = []
    true_next_state_gps = []
    next_state_funcs = []
    true_next_state_funcs = []
    next_state_jacobians = []
    state_var_funcs = []
    stage_cost_funcs = []
    stage_cost_jacobians = []

    for d, func in enumerate(unknown_stage_cost_functions):
        if use_gp and func['use_gp']:

            # define cost function GP approximation
            cost_gp = StageCostGP(model.devices[d],
                                  func['state_cols'] + func['input_cols'],
                                  func['output_variance'],
                                  func['meas_noise'],
                                  func['length_scale'],
                                  func['sampling_t_step'],
                                  stage_cost_max_n_training_samples, stage_cost_n_init_training_samples,
                                  n_test_samples, model.devices[d].n_states,
                                  model.devices[d].n_inputs, model.devices[d].n_disturbances)

            func['max_n_training_samples'] = stage_cost_max_n_training_samples
            func['n_init_training_samples'] = stage_cost_n_init_training_samples

            stage_cost_gps.append(cost_gp)

            # if modeling the difference between the observed function and the prior
            def stage_cost_func(z_stage, gp=cost_gp, device_idx=d, return_std=False):
                # first 0 index fetches the mean, second 0 index fetches the first row of the return predictions
                stage_cost = gp.predict(z_stage[np.newaxis, :])

                if return_std:
                    return model.devices[device_idx].stage_cost_prior_func(z_stage) \
                           + stage_cost[0][0], stage_cost[1][0]
                else:
                    return model.devices[device_idx].stage_cost_prior_func(z_stage) \
                           + stage_cost[0][0]

            def stage_cost_jacobian(z_stage, gp=cost_gp, device_idx=d):
                # pred = lambda z: gp.predict(z)[0][0]
                # true_jac = jacobian_func(pred, z_stage)
                prior_jac = model.devices[device_idx].stage_cost_prior_jacobian(z_stage)
                calc_jac = gp.mean_jacobian(z_stage[np.newaxis, :])
                # delta = true_jac - variation_jac
                return prior_jac + calc_jac

        else:
            stage_cost_gps.append(None)

            stage_cost_func = model.devices[d].true_stage_cost_func
            stage_cost_jacobian = model.devices[d].true_stage_cost_jacobian

        stage_cost_funcs.append(stage_cost_func)
        stage_cost_jacobians.append(stage_cost_jacobian)

    def stage_cost_func(z_stage, k=0):
        # first 0 index fetches the mean, second 0 index fetches the first row of the return predictions
        model.set_simulation_step(k)
        stage_cost = model.system_stage_cost_func(z_stage)
        for d, dev_func in enumerate(stage_cost_funcs):
            stage_cost = stage_cost + dev_func(z_stage[model.device_zstage_indices[d]])
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

    def device_stage_cost_func(z_stage, k=0, return_std=True):
        # first 0 index fetches the mean, second 0 index fetches the first row of the return predictions
        model.set_simulation_step(k)
        stage_cost_pred = []
        stage_cost_std = []
        for d, dev_func in enumerate(stage_cost_funcs):
            stage_cost = dev_func(z_stage[model.device_zstage_indices[d]], return_std=return_std)
            if return_std:
                stage_cost_pred = stage_cost_pred + list(stage_cost[0])
                stage_cost_std = stage_cost_std + list(stage_cost[1])
            else:
                stage_cost_pred = stage_cost_pred + list(stage_cost)

        if return_std:
            return stage_cost_pred, stage_cost_std
        else:
            return stage_cost_pred

    for d, funcs in enumerate(unknown_next_state_functions):
        state_var_funcs.append([])
        next_state_funcs.append([])
        true_next_state_funcs.append([])
        next_state_jacobians.append([])
        for dim, func in enumerate(funcs):
            if use_gp and func['use_gp']:
                next_state_gp = NextStateGP(model.devices[d], dim,
                                            func['state_cols'] + func['input_cols'] + func['disturbance_cols'],
                                            func['output_variance'],
                                            func['meas_noise'],
                                            func['length_scale'],
                                            func['sampling_t_step'],
                                            next_state_max_n_training_samples,
                                            next_state_n_init_training_samples, n_test_samples,
                                            state_lag, input_lag, disturbance_lag, output_lag,
                                            model.devices[d].n_states, model.devices[d].n_inputs,
                                            model.devices[d].n_disturbances, model.devices[d].n_outputs,
                                            len(func['state_cols'] + func['input_cols'] + func['disturbance_cols']),
                                            is_true_system=False)

                # next_state_gp.prior_mean = 25 #model.devices[d].init_state

                next_state_gps.append(next_state_gp)
                func['max_n_training_samples'] = next_state_max_n_training_samples
                func['n_init_training_samples'] = next_state_n_init_training_samples

                # modeling the difference between the observed function and the prior
                def state_var_func(z_lagged, gp=next_state_gp, return_std=False):
                    post_mean, post_std = gp.predict(z_lagged[np.newaxis, :])

                    if return_std:
                        return (post_mean[0][0], post_std[0][0])
                    else:
                        return post_mean[0][0]

                # call conditional probability calculator each time the function is needs
                def next_state_func(z_lagged, device_idx=d, output_dim=dim, return_std=False):

                    state_var = state_var_func(z_lagged, return_std=return_std)
                    state = model.devices[device_idx].next_state_prior_func(z_lagged)[output_dim]

                    if return_std:
                        return (state + state_var[0], state_var[1])
                    else:
                        return state + state_var

                def next_state_jacobian(z_lagged, gp=next_state_gp, device_idx=d, output_dim=dim):
                    prior_jac = model.devices[device_idx].next_state_prior_jacobian(z_lagged)[output_dim]
                    post_mean_jac = gp.mean_jacobian(z_lagged[np.newaxis, :])
                    jac_var = post_mean_jac
                    return prior_jac + jac_var

                if not func['synthetic_data']:
                    true_sys_func = dict(func)
                    true_sys_func['n_init_training_samples'] = input_params['true_sys_n_training_samples']
                    true_sys_func['max_n_training_samples'] = input_params['true_sys_n_training_samples']
                    true_sys_func['sampling_t_step'] = 1e6
                    true_next_state_gp = NextStateGP(model.devices[d], dim,
                                                     true_sys_func['state_cols'] + true_sys_func['input_cols']
                                                     + true_sys_func['disturbance_cols'],
                                                     true_sys_func['output_variance'],
                                                     true_sys_func['meas_noise'],
                                                     true_sys_func['length_scale'],
                                                     true_sys_func['sampling_t_step'],
                                                     true_sys_func['max_n_training_samples'],
                                                     true_sys_func['n_init_training_samples'], n_test_samples,
                                                     state_lag, input_lag, disturbance_lag, output_lag,
                                                     model.devices[d].n_states, model.devices[d].n_inputs,
                                                     model.devices[d].n_disturbances, model.devices[d].n_outputs,
                                                     len(true_sys_func['state_cols'] + true_sys_func['input_cols']
                                                         + true_sys_func['disturbance_cols']),
                                                     is_true_system=True)
                    # true_next_state_gp.prior_mean = 25#model.devices[d].init_state

                    true_next_state_gps.append(true_next_state_gp)

                    def true_state_var_func(z_lagged, gp=true_next_state_gp, return_std=False):
                        post_mean, post_std = gp.predict(z_lagged[np.newaxis, :])

                        if return_std:
                            return (post_mean[0][0], post_std[0][0])
                        else:
                            return post_mean[0][0]

                    def true_next_state_func(z_lagged, gp=true_next_state_gp, device_idx=d, output_dim=dim,
                                             return_std=False):

                        state_var = true_state_var_func(z_lagged, gp=gp, return_std=return_std)
                        state = model.devices[device_idx].next_state_prior_func(z_lagged)[output_dim]

                        if return_std:
                            return (state + state_var[0], state_var[1])
                        else:
                            return state + state_var

                    true_next_state_funcs[d].append(true_next_state_func)
                else:
                    true_next_state_gps.append(None)
                    true_next_state_funcs[d].append(None)

            else:
                next_state_gps.append(None)
                true_next_state_gps.append(None)

                def state_var_func(z_lagged, return_std=False):
                    if return_std:
                        return (0, 0)
                    else:
                        return 0

                # bind d variable as a default parameter to function so that it doesn't always refer to last possible state
                def next_state_func(z_lagged, device_idx=d, output_dim=dim, return_std=False):
                    if return_std:
                        return (model.devices[device_idx].true_next_state_func(z_lagged)[output_dim], 0)
                    else:
                        return model.devices[device_idx].true_next_state_func(z_lagged)[output_dim]

                def next_state_jacobian(z_lagged, device_idx=d, output_dim=dim):
                    return jacobian_func(model.devices[device_idx].true_next_state_func, z_lagged)[output_dim]

                true_next_state_func = next_state_func

            state_var_funcs[d].append(state_var_func)
            next_state_funcs[d].append(next_state_func)
            next_state_jacobians[d].append(next_state_jacobian)

    ####################################################################################################################

    model.state_var_funcs = state_var_funcs
    model.next_state_funcs = next_state_funcs
    model.true_next_state_funcs = true_next_state_funcs
    model.next_state_jacobians = next_state_jacobians

    ####################################################################################################################

    # INITIALISE OPTIMAL CONTROL PROBLEM
    ocp = OCP(n_horizon=n_horizon, state_lag=state_lag, input_lag=input_lag, disturbance_lag=disturbance_lag,
              output_lag=output_lag, mpc_t_step=mpc_t_step, imp_bounds_only=imp_bounds_only,
              model_type=model_type)

    # INITIALISE DYNAMIC-STATE AND COST FUNCTIONS, EXPLICIT AND IMPLICIT MPC CONSTRAINTS IN OCP

    ocp.set_cost_funcs(stage_cost_func, terminal_cost_func, device_stage_cost_func)
    ocp.set_cost_jacobians(stage_cost_jacobian, terminal_cost_jacobian)

    ocp.set_next_state_func(model.next_state_func)
    ocp.set_next_state_jacobian(model.next_state_jacobian)

    # set explicit state inequality constraints
    ocp.set_stage_ineq_constraint_func(model.stage_ineq_constraint_func)
    ocp.set_stage_ineq_constraint_jacobian(model.stage_ineq_constraint_jacobian)

    # set implicit constraints/feasible sets for x and u at each time-step
    ocp.set_imp_constraints(stage_constraint_func=model.stage_constraint_func,
                            term_constraint_func=model.term_constraint_func)
    ocp.set_imp_bounds(model.stage_bounds, model.term_bounds)

    ocp.set_opt_vars(model.n_states, model.n_inputs, model.n_disturbances, model.n_outputs)
    ocp.set_params(model.parameters)

    ocp.set_constraints()

    ####################################################################################################################

    # STORE TRAINING DATA

    # instantiate a TrainingDataReader object to generate system measurements
    data_reader = DataReader(input_params, model.n_states, model.n_inputs, model.n_disturbances,
                             model.n_outputs, n_simulation_steps)
    gp_df_cols = ['x_train', 'y_train']
    stage_cost_training_data_dfs = []
    next_state_training_data_dfs = []

    if run_mpc or run_cost_gp:
        for d, func in enumerate(unknown_stage_cost_functions):
            if use_gp and func['use_gp']:

                if func['synthetic_data']:
                    # get input and output synthesized training data
                    x_train, x_train_ind, y_true, y_train, y_train_ind \
                        = data_reader.generate_training_data(model.devices[d], func, mpc_t_step,
                                                             model.devices[d].true_stage_cost_func,
                                                             model.devices[d].stage_cost_prior_func, is_state_gp=False,
                                                             output_dim=0, run_mpc=run_mpc)

                    plot_independent_data = not run_mpc
                    if plot_independent_data:
                        x_train = x_train_ind
                        y_train = y_train_ind
                        stage_cost_gps[d].max_n_training_samples *= x_train.shape[1]
                        stage_cost_gps[d].n_init_training_samples *= x_train.shape[1]
                        n_training_indices = func['max_n_training_samples'] * x_train.shape[1]
                    else:
                        n_training_indices = func['n_init_training_samples']
                        # + int(n_simulation_steps * (mpc_t_step / func['sampling_t_step']))
                    training_indices = np.arange(n_training_indices)
                else:
                    # get input and output true training data
                    training_indices, x_train, y_train, test_indices, x_test, y_true \
                        = data_reader.read_training_data(func, run_cost_gp, mpc_t_step, test_offset)
                    stage_cost_gps[d].test_indices = test_indices
                    stage_cost_gps[d].x_test = x_test
                    stage_cost_gps[d].y_true = y_true

                    # add cost training data to database
                cost_df = pd.DataFrame(columns=gp_df_cols)
                cost_df = data_reader.add_training_data(cost_df, training_indices, x_train, y_train, is_init=True,
                                                        n_init_training_samples=stage_cost_gps[d].n_init_training_samples)

                stage_cost_training_data_dfs.append(cost_df)
            else:
                stage_cost_training_data_dfs.append(None)

    if run_mpc or run_state_gp:
        # for each state of each device
        f = -1
        for d, funcs in enumerate(unknown_next_state_functions):
            for dim, func in enumerate(funcs):
                f += 1
                # if using a gp to estimate this state
                if use_gp and func['use_gp']:
                    # if there is true training data

                    if func['synthetic_data']:
                        # get input and output synthesized training data
                        x_train, x_train_ind, y_true, y_train, y_train_ind \
                            = data_reader.generate_training_data(
                            model.devices[d], func, mpc_t_step, model.devices[d].true_next_state_func,
                            model.devices[d].next_state_prior_func, is_state_gp=True, output_dim=dim, run_mpc=run_mpc)

                        plot_independent_data = not run_mpc
                        if plot_independent_data:
                            x_train = x_train_ind
                            y_train = y_train_ind
                            next_state_gps[f].max_n_training_samples *= x_train.shape[1]
                            next_state_gps[f].n_init_training_samples *= x_train.shape[1]
                            n_training_indices = func['max_n_training_samples'] * x_train.shape[1]
                        else:
                            n_training_indices = func['n_init_training_samples']
                            # + int(n_simulation_steps * (mpc_t_step / func['sampling_t_step']))

                        training_indices = np.arange(n_training_indices)
                    else:
                        # get input and output true training data
                        training_indices, x_train, y_train, test_indices, x_test, y_true = \
                            data_reader.read_training_data(func, run_state_gp, mpc_t_step, test_offset,
                                                           zero_prior=zero_prior)
                        next_state_gps[f].test_indices = test_indices
                        next_state_gps[f].x_test = x_test
                        next_state_gps[f].y_true = y_true

                        true_sys_func = dict(func)
                        true_sys_func['n_init_training_samples'] = input_params['true_sys_n_training_samples']
                        true_sys_func['max_n_training_samples'] = input_params['true_sys_n_training_samples']
                        true_sys_func['sampling_t_step'] = 1e6

                        true_sys_training_indices, true_sys_x_train, true_sys_y_train, true_sys_test_indices, \
                        true_sys_x_test, true_sys_y_true \
                            = data_reader.read_training_data(true_sys_func, True, mpc_t_step, test_offset,
                                                             zero_prior=zero_prior)

                        true_next_state_gps[f].test_indices = true_sys_test_indices
                        true_next_state_gps[f].training_indices = true_sys_training_indices
                        true_next_state_gps[f].x_test = true_sys_x_test
                        true_next_state_gps[f].y_true = true_sys_y_true
                        true_next_state_gps[f].x_train = true_sys_x_train
                        true_next_state_gps[f].y_train = true_sys_y_train

                    # add to database
                    next_state_df = pd.DataFrame(columns=gp_df_cols)
                    next_state_df = data_reader.add_training_data(next_state_df, training_indices, x_train, y_train,
                                                                  is_init=True,
                                                                  n_init_training_samples=
                                                                  next_state_gps[f].n_init_training_samples)

                    # add next state (error) training data to database
                    next_state_training_data_dfs.append(next_state_df)

                else:
                    next_state_training_data_dfs.append(None)

    ####################################################################################################################

    # FETCH INITIAL SAMPLES OF DYNAMIC STATE AND COST FUNCTIONS, OPTIMIZE HYPERPARAMETERS AND TEST

    # collect function samples from DataFrame and add to GP training set
    # train gps with independent data (ie only a single feature nonzero for each sample) if using synthetic data
    if run_mpc or run_state_gp:
        f = -1
        for d, device_funcs in enumerate(unknown_next_state_functions):
            for dim, state_func in enumerate(device_funcs):
                f += 1
                if use_gp and state_func['use_gp']:

                    for g, gp in enumerate([next_state_gps[f], true_next_state_gps[f]]):
                        if gp is None:
                            continue
                        if g == 0:
                            next_state_training_data_dfs[f] = pd.DataFrame(
                                gp.collect_training_data(next_state_training_data_dfs[f],
                                                         n_samples=next_state_gps[f].n_init_training_samples,
                                                         is_init=True))
                        else:
                            pass
                        # TODO

                        # ref_weight = gp.y_train.shape[0]
                        # gp.prior_mean = np.mean(np.concatenate([model.devices[d].ref_state[dim] * np.ones(ref_weight),
                        #                          gp.y_train[:, gp.output_dim]]))
                        # print(model.devices[d].ref_state[dim], np.mean(gp.y_train[:, gp.output_dim]), gp.prior_mean)
                        gp.update_device_bounds(0, n_horizon)
                        if g == 0:
                            gp.calculate_opt_hyperparams()
                        gp.update_inv_cov_train()
                        if not run_mpc and g == 0:
                            gp.test(data_reader, model.devices[d], state_func, n_test_samples, n_simulation_steps,
                                    next_state_gps[f].max_n_training_samples, mpc_t_step, sub_results_dirs[-1],
                                    simulation_dir, simulation_name, run_mpc, use_linear_test_values, d, dim)

    if run_mpc or run_cost_gp:
        for f, func in enumerate(unknown_stage_cost_functions):
            if use_gp and func['use_gp']:
                stage_cost_training_data_dfs[f] = pd.DataFrame(
                    stage_cost_gps[f].collect_training_data(stage_cost_training_data_dfs[f],
                                                            n_samples=stage_cost_gps[f].n_init_training_samples,
                                                            is_init=True))
                # stage_cost_gps[f].update_device_bounds(0, n_horizon)
                # stage_cost_gps[f].calculate_opt_hyperparams()
                stage_cost_gps[f].update_inv_cov_train()
                if not run_mpc:
                    stage_cost_gps[f].test(data_reader, model.devices[f], func, n_test_samples, n_simulation_steps,
                                           stage_cost_gps[f].max_n_training_samples, mpc_t_step, sub_results_dirs[-1],
                                           simulation_dir,
                                           simulation_name, run_mpc, use_linear_test_values, f, 0)

    ####################################################################################################################

    # INITIALIZE PD OPTIMIZER
    pdo = PDOptimizationMethod(ocp=ocp)

    ####################################################################################################################

    # INITIALIZE MPC
    mpc = MPC(ocp, pdo, opt_params, n_horizon, use_opt_method=use_opt_method)

    ####################################################################################################################
    # SIMULATE SYSTEM

    if run_mpc:
        mpc_inputs = {key: [input_params[key]]
                      for key in ['simulation_name', 'maxiter', 'alpha', 'use_gp',
                                  'eta_ineq', 'eta_eq', 'eps', 'n_horizon', 'n_simulation_steps']}
        if not os.path.exists(f'{results_dir}/mpc_results.csv'):
            mpc_results_df = pd.DataFrame(mpc_inputs)
        else:
            mpc_results_df = pd.read_csv(f'{results_dir}/mpc_results.csv', engine='python', index_col=0)
            existing_row_indices = (mpc_results_df['simulation_name'] == simulation_name)
            # mpc_results_df = mpc_results_df.loc[~existing_row_indices]

            if existing_row_indices.any(axis=0):
                mpc_results_df = mpc_results_df.loc[~existing_row_indices]

            mpc_results_df = mpc_results_df.append(mpc_inputs, ignore_index=True)
            mpc_results_df = mpc_results_df.reset_index(drop=True)

        mpc_results_df.to_csv(f'{results_dir}/mpc_results.csv')

        simulator = Simulator(ocp, mpc, model, data_reader, n_simulation_steps)

        if input_params['plot_existing_data'] \
                and os.path.exists(f'{sub_results_dirs[-1]}/trajectory.csv'):
            # fetch mpc trajectory data
            traj_df = pd.read_csv(f'{sub_results_dirs[-1]}/trajectory.csv', index_col=0)
            traj_df = traj_df.applymap(
                lambda cell: np.asarray([g[0] for g in re.findall('(-*\d*\.\d*(e-*\d+)*)+', cell)]).astype(float))
            simulator.set_trajectories(traj_df)
            # idx = np.arange(259, 268)
            # # TODO check variance threshold leading up to peak
            # x_test = np.vstack([simulator.state_trajectory[idx, 3], simulator.input_trajectory[idx, 3],
            #                     simulator.disturbance_trajectory[idx, 3]]).T
            # next_state_gps[3].predict(x_test)

            # conv_df = pd.read_csv(f'')
            # conv_df = conv_df.applymap(
            #     lambda cell: np.asarray([g[0] for g in re.findall('(-*\d*\.\d*(e-*\d+)*)+', cell)]).astype(float))
            # mpc.opt_object.set_trajectories(traj_df)
        else:
            true_states, true_disturbances = data_reader.read_true_data(n_horizon, unknown_next_state_functions,
                                                                        mpc_t_step)
            if true_disturbances.shape[1] == 0:
                for device in model.devices:
                    true_disturbances = np.hstack([true_disturbances, device.disturbance_train])
                # 30 * np.ones((n_simulation_steps + n_horizon, 1))
            for d, device in enumerate(model.devices):
                for s, state in enumerate(unknown_next_state_functions[d]):
                    if 'training_data_path' in state and state['training_data_path'] is not None:
                        device.init_state[s] \
                            = true_states[0, model.device_xstage_indices[d][s]]
            model.set_init_state()

            true_states = None

            simulator.simulate(model.init_state, true_states, true_disturbances,
                               stage_cost_gps, stage_cost_training_data_dfs,
                               next_state_gps, next_state_training_data_dfs,
                               true_next_state_gps=true_next_state_gps,
                               synthetic_state_data=np.any(
                                   [[state['synthetic_data'] if 'synthetic_data' in state else False
                                     for state in device]
                                    for device in unknown_next_state_functions]))

        system_trajectory = [simulator.state_trajectory, simulator.input_trajectory, simulator.disturbance_trajectory,
                             simulator.cost_trajectory,
                             simulator.ineq_constraint_trajectory, simulator.eq_constraint_trajectory,
                             simulator.ineq_dual_trajectory, simulator.eq_dual_trajectory]

        sim_df_cols = ['x0', 'u0', 'w0', 'cost', 'ineq_constraints', 'eq_constraints', 'ineq_dual_vars', 'eq_dual_vars']

        # export trajectories
        simulation_dict = {key: system_trajectory[sim_df_cols.index(key)].tolist() for key in sim_df_cols}
        simulation_df = pd.DataFrame(simulation_dict)
        simulation_df.to_csv(f'{sub_results_dirs[-1]}/trajectory.csv', index_label='time_step')

        # export figures
        # traj_fig, error_fig, dual_fig = simulator.plot_trajectory(comp_sim_df, bounds=trajectory_bounds,
        #                                                           return_tracking_error=True)
        # conv_fig, conv_ax, var_fig, var_ax = simulator.plot_convergence()
        # traj_fig.savefig(f'{sub_results_dir}/traj_plot')
        # error_fig.savefig(f'{sub_results_dir}/error_plot')
        # conv_fig.savefig(f'{sub_results_dir}/conv_plot')
        # var_fig.savefig(f'{sub_results_dir}/var_plot')

        if n_inverted_pendulums == 1:
            writer = animation.FFMpegWriter(fps=15)
            anim_fig, anim_ax, anim = simulator.animate()
            anim_fig.savefig(f'{sub_results_dirs[-1]}/anim_plot')
            anim.save(f'{sub_results_dirs[-1]}/pendulum.mp4', writer=writer)

        ####################################################################################################################

    # COMPARE 2 SIMULATIONS
    if compare_simulations and len(simulation_comparisons) == 2:
        sim_a_name = simulation_comparisons[0]
        sim_b_name = simulation_comparisons[1]
        df_a = pd.read_csv(sim_a_name + '/' + 'trajectory.csv', engine="python", index_col=0)
        df_b = pd.read_csv(sim_b_name + '/' + 'trajectory.csv', engine="python", index_col=0)
        simulation_comparator = SimulationComparator(df_a, df_b)
        comp_fig, comp_ax = simulation_comparator.plot_rel_error()

        comp_fig.savefig(f'{compared_sims_dir}/traj_plot')

    if run_mpc:
        return simulator
    elif run_state_gp and run_cost_gp:
        return [gp for gp in next_state_gps if gp is not None] + [gp for gp in stage_cost_gps if gp is not None]
    elif run_state_gp:
        return [gp for gp in next_state_gps if gp is not None]
    elif run_cost_gp:
        return [gp for gp in stage_cost_gps if gp is not None]


if __name__ == '__main__':

    # define experiment and system input parameters, measurement vector consists of states and cost

    RESET = False

    simulation_dir = f'./{default_input_params["simulation_dir"]}'
    results_dir = f'{simulation_dir}/results'
    training_data_dir = f'{simulation_dir}/training_data'
    compared_sims_dir = f'{simulation_dir}/results/Compared Simulations'
    export_dir = f'{simulation_dir}/exports'
    dirs = [simulation_dir, results_dir, training_data_dir, compared_sims_dir, export_dir]

    if RESET:
        shutil.rmtree(simulation_dir)

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # run online TCL, offƒline IP and offline TCL simulations
    sim_indices = [6, 7, 8]
    for sim_batch in sim_indices:
        sim_batch_data = []
        gp_batch_data = []
        sub_results_dirs = []
        for custom_input_params in np.array(simulations)[sim_batch]:
            sub_results_dirs.append(f'{results_dir}/{custom_input_params["simulation_name"]}')

            if 'num_devices' in custom_input_params:
                num_devices = custom_input_params['num_devices']

            if not os.path.exists(sub_results_dirs[-1]):
                os.mkdir(sub_results_dirs[-1])
            # elif not default_input_params['plot_existing_data']:
            #     for _, _, files in os.walk(sub_results_dirs[-1]):
            #         for file in files:
            #             os.remove(f'{sub_results_dirs[-1]}/{file}')

            new_input_params = {
                key: custom_input_params[key] if key in custom_input_params else default_input_params[key]
                for key in default_input_params}
            new_input_params['stage_cost_funcs'] = stage_cost_funcs
            new_input_params['next_state_funcs'] = next_state_funcs

            # list of simulation or gp objects
            res = main(new_input_params)
            if new_input_params['run_mpc']:
                sim_batch_data.append(res)
            else:
                gp_batch_data.append(res)

            trajectory_bounds = {'x0': [None, None], 'u0': [None, None], 'cost': [None, None],
                                 'eq_con': [None, None], 'ineq_con': [None, None]}
        for simulator in sim_batch_data:
            min_state = np.min(simulator.state_trajectory)
            max_state = np.max(simulator.state_trajectory)
            min_input = np.min(simulator.input_trajectory)
            max_input = np.max(simulator.input_trajectory)
            min_cost = np.min(simulator.cost_trajectory)
            max_cost = np.max(simulator.cost_trajectory)
            min_eq_con = np.min(simulator.eq_constraint_trajectory)
            max_eq_con = np.max(simulator.eq_constraint_trajectory)
            if simulator.ineq_constraint_trajectory.shape[1]:
                min_ineq_con = np.min(simulator.ineq_constraint_trajectory)
                max_ineq_con = np.max(simulator.ineq_constraint_trajectory)
            else:
                min_ineq_con = 0
                max_ineq_con = 0

            bounds = {'x0': [min_state, max_state], 'u0': [min_input, max_input], 'cost': [min_cost, max_cost],
                      'eq_con': [min_eq_con, max_eq_con], 'ineq_con': [min_ineq_con, max_ineq_con]}

            for key, val in trajectory_bounds.items():
                if val[0] is None or bounds[key][0] < val[0]:
                    trajectory_bounds[key][0] = val[0]

                if val[1] is None or bounds[key][1] > val[1]:
                    trajectory_bounds[key][1] = val[1]

        for s, simulator in enumerate(sim_batch_data):

            if new_input_params['run_mpc']:
                # if running an online algorithm
                if False and new_input_params['maxiter'] <= 50:
                    # fetch equivalent offline simulation
                    # use_gp = use_gp and (any([func["use_gp"] for func in unknown_stage_cost_functions]) \
                    #                      or any(np.array([[func['use_gp'] for func in device_funcs]
                    #                                       for device_funcs in unknown_next_state_functions]).flatten()))
                    # comp_sim_name = f'{"GP" if use_gp else "Known Models"} Offline N={n_horizon}'
                    comp_sim_name = new_input_params['simulation_name'].replace('Online', 'Offline')
                    comp_results_dir = f'{results_dir}/{comp_sim_name}/trajectory.csv'
                    comp_sim_df = pd.read_csv(comp_results_dir)

                    sim_df_cols = ['x0', 'u0', 'w0', 'cost', 'ineq_constraints', 'eq_constraints', 'ineq_dual_vars',
                                   'eq_dual_vars']
                    comp_sim_df[sim_df_cols] = comp_sim_df[sim_df_cols].applymap(
                        lambda cell: np.asarray([g[0] for g in re.findall('(-*\d*\.\d*(e-*\d+)*)+', cell)]).astype(
                            float))
                    if len(comp_sim_df.index) != new_input_params['n_simulation_steps']:
                        comp_sim_df = None
                else:
                    comp_sim_df = None

                traj_fig, error_fig, dual_fig = simulator.plot_trajectory(comp_sim_df,
                                                                          bounds=list(trajectory_bounds.values()),
                                                                          return_tracking_error=True)
                conv_fig, conv_ax, var_fig, var_ax = simulator.plot_convergence()
                traj_fig.savefig(f'{sub_results_dirs[s]}/traj_plot')
                error_fig.savefig(f'{sub_results_dirs[s]}/error_plot')
                conv_fig.savefig(f'{sub_results_dirs[s]}/conv_plot')
                var_fig.savefig(f'{sub_results_dirs[s]}/var_plot')

        for g, gp in enumerate(np.array(gp_batch_data).flatten()):
            for ff, fig in enumerate(gp.prediction_figs):
                fig.show()
                fig.savefig(f'{sub_results_dirs[g]}/{gp.prediction_fig_filename}_{ff}')

            gp.score_fig.savefig(f'{sub_results_dirs[g]}/{gp.prediction_fig_filename}_score')

    generate_exports = GenerateExports(export_dir, results_dir)
    generate_exports.copy_figures()
    generate_exports.generate_gp_params_table()
    generate_exports.generate_mpc_params_table()
