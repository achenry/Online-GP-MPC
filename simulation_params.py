import numpy as np

num_devices = {'tcl': 5, 'tcl2': 0, 'batt': 0, 'ip': 1}
stage_cost_funcs = {'tcl': [{'use_gp': True,
                             'training_data_path': f'./device_data/tcl_{d + 1}_cost_data.csv',
                             # temperature outside building, temperature, solar irradiance,
                             # internal irradiance, mass air flow, discharge air temperature
                             'state_cols': [f'T_zone_{d + 1}'], 'input_cols': [f'P_zone_{d + 1}'],
                             # [f'ms_dot_{d + 1}', f'T_da_{d + 1}'],
                             'disturbance_cols': ['T_outside'],  # , f'Q_solar_{d + 1}', f'Q_internal_{d + 1}'],
                             'sampling_t_step': 60 * 60,  # * 60,
                             'length_scale': [8.0, 1e6, 1e6],
                             'output_variance': 50,
                             'meas_noise': 1e-6,
                             'model_error': True,
                             'function_type': 'stage_cost',
                             'synthetic_data': True,
                             'input_labels': [f'T_zone_{d + 1}']}
                            for d in range(num_devices['tcl'])],

                    'tcl2': [{'use_gp': True,
                              'training_data_path': None,
                              'state_cols': [f'T_zone_{d + 1}'], 'input_cols': [f'P_zone_{d + 1}'],
                              # [f'ms_dot_{d + 1}', f'T_da_{d + 1}'],
                              'disturbance_cols': ['T_outside'],
                              'sampling_t_step': 60 * 60,  # * 60,
                              'length_scale': [8.0, 1e6, 1e6],
                              'output_variance': 50,
                              'meas_noise': 1e-6,
                              'model_error': True,
                              'function_type': 'stage_cost',
                              'synthetic_data': True,
                              'input_labels': [f'T_zone_{d + 1}']}
                             for d in range(num_devices['tcl2'])],

                    'batt': [{'use_gp': True,
                              'training_data_path': None,  # f'./device_data/battery_{d + 1}_cost_data.csv',
                              'state_cols': [f'soc_{d + 1}'], 'input_cols': [f'E_out_{d + 1}'],
                              'disturbance_cols': [f'P_ref_{d + 1}'],
                              'sampling_t_step': 5,
                              'length_scale': [1e5, 16000 * (5 / 60), 16000],
                              'output_variance': 2 * 1e8,
                              'meas_noise': 0.01,
                              'model_error': True,
                              'function_type': 'stage_cost',
                              'synthetic_data': True,
                              'input_labels': [f'E_out_{d + 1}', f'P_ref_{d + 1}']}
                             for d in range(num_devices['batt'])],

                    'ip': [{'use_gp': True,
                            'training_data_path': None,
                            'state_cols': [f'x1_{d + 1}', f'x2_{d + 1}'],
                            'input_cols': [f'u_{d + 1}'],
                            'disturbance_cols': [f'w_{d + 1}'],
                            'sampling_t_step': 0.1,
                            'length_scale': [4 * np.pi, 1e6, 1e6, 1e6],
                            'output_variance': 50,
                            'meas_noise': 1e-6,
                            'model_error': True,
                            'function_type': 'stage_cost',
                            'synthetic_data': True,
                            'input_labels': [f'x1_{d + 1}']}
                           for d in range(num_devices['ip'])
                           ]}

# tcl_length_scales = [[1.961970139642346, 7.728661224895884, 7.001233790723497]]
# tcl_output_variances = [15.630980809394867]

tcl_length_scales = [[1.961970139642346, 7.728661224895884, 7.001233790723497],
                     [2, 11.1797391036531, 8.90963359214152],
                     [11.1254769545664, 7.381673770784, 3.69119685757961],
                     # [25.4234080197009, 2.01519147113952, 2.90914994786139],
                     [2, 11.1797391036531, 8.90963359214152],
                     [7.63607224455928, 9.66447201102184, 6.23435408073675]]  # 4.--
tcl_output_variances = [15.630980809394867, 15.1164105644889, 22.8059090574877, 25.0911884588030, 19.4754021007873]
# tcl_length_scales = [[0.89224197, 2.44347036, 1.68678906],
#                      [1.18110114, 1.70193122, 2.44347036],
#                      [2.4092377, 1.99900041, 1.30595076],
#                      [0.69314718, 2.41410313, 2.18713312],
#                      [2.03288337, 2.26845648, 1.83007498]]
# tcl_output_variances = np.array([3.64894107, 3.53302678, 3.12701967, 3.22251673, 2.96915224]) ** 2
tcl_noise = [3.8351e-04, 0.0224, 0.0125, 8.8813e-04, 1.5219e-04]

next_state_funcs = {'tcl': [[{'use_gp': True,
                              'training_data_path': f'./device_data/tcl_{d + 1}_data.csv',
                              'state_cols': [f'T_zone_{d + 1}'], 'input_cols': [f'P_zone_{d + 1}'],
                              # [f'ms_dot_{d + 1}', 'T_da'],
                              'disturbance_cols': ['T_outside'],  # f'Q_solar_{d + 1}', f'Q_internal_{d + 1}'],
                              'output_state_col': [f'T_zone_{d + 1}'],
                              'sampling_t_step': 60 * 5,  # * 60,
                              'length_scale': tcl_length_scales[d],
                              'output_variance': tcl_output_variances[d],
                              'meas_noise': tcl_noise[d],
                              'model_error': True,
                              'function_type': 'state',
                              'synthetic_data': False,
                              'input_labels': [f'T_zone_{d + 1}', f'P_zone_{d + 1}', f'T_outside'],
                              # , f'm_dot_a_{d + 1}', 'T_a'],
                              'output_labels': [f'T_zone_{d + 1}']}]
                            for d in range(num_devices['tcl'])],

                    'tcl2': [[{'use_gp': True,
                               'training_data_path':  f'./device_data/tcl_{d + 1}_data.csv',
                               'state_cols': [f'T_zone_{d + 1}'], 'input_cols': [f'P_zone_{d + 1}'],
                               # [f'ms_dot_{d + 1}', 'T_da'],
                               'disturbance_cols': ['T_outside'],  # f'Q_solar_{d + 1}', f'Q_internal_{d + 1}'],
                               'output_state_col': [f'T_zone_{d + 1}'],
                               'sampling_t_step': 60,  # * 60,
                               'length_scale': tcl_length_scales[d],
                               'output_variance': tcl_output_variances[d],
                               'meas_noise': tcl_noise[d],
                               'model_error': True,
                               'function_type': 'state',
                               'synthetic_data': True,
                               'input_labels': [f'T_zone_{d + 1}', f'P_zone_{d + 1}', f'T_outside'],
                               # , f'm_dot_a_{d + 1}', 'T_a'],
                               'output_labels': [f'T_zone_{d + 1}']}]
                             for d in range(num_devices['tcl2'])],

                    'batt': [[{'use_gp': True,
                               'training_data_path': None,  # f'./device_data/battery_{d + 1}_data.csv',
                               'state_cols': [f'soc_{d + 1}'], 'input_cols': [f'E_out_{d + 1}'],
                               'disturbance_cols': [f'P_ref_{d + 1}'],
                               'output_state_col': [f'soc_{d + 1}'],
                               'sampling_t_step': 5,  # * 60,
                               'length_scale': [0.001, 1., 100],
                               'output_variance': 0.01,
                               'meas_noise': 1e-6,
                               'model_error': True,
                               'function_type': 'state',
                               'synthetic_data': True,
                               'input_labels': [f'soc_{d + 1}', f'E_out_{d + 1}'],
                               'output_labels': [f'soc_{d + 1}']}]
                             for d in range(num_devices['batt'])],

                    'ip': [[{'use_gp': False},
                            {'use_gp': True,
                             'training_data_path': None,
                             'state_cols': [f'theta_{d + 1}', f'theta_dot_{d + 1}'],
                             'input_cols': [f'u_{d + 1}'],
                             'disturbance_cols': [f'w_{d + 1}'],
                             'output_state_col': [f'theta_dot_{d + 1}'],
                             'sampling_t_step': 0.1,
                             'length_scale': [np.pi, 100, 100, 100],
                             'output_variance': 10,
                             'meas_noise': 1e-6,
                             'model_error': True,
                             'function_type': 'state',
                             'synthetic_data': True,
                             'input_labels': [f'theta_{d + 1}', f'theta_dot_{d + 1}', f'u_{d + 1}'],
                             'output_labels': [f'theta_dot_{d + 1}']}]
                           for d in range(num_devices['ip'])]}

default_input_params = {'plot_existing_data': False, 'init_maxiter': 2000,
                        'n_test_samples': 48, 'test_offset': 12, # None
                        'n_simulation_steps': 288, 'n_horizon': 12, 'mpc_t_step': 60 * 5, 'R': 0,
                        'state_lag': 0, 'input_lag': 0, 'disturbance_lag': 0, 'output_lag': 0,
                        'stage_cost_n_init_training_samples': 25, 'next_state_n_init_training_samples': 100,
                        'stage_cost_max_n_training_samples': 50, 'next_state_max_n_training_samples': 200,
                        'true_sys_n_training_samples': 6 * 288 + 287,
                        'use_gp': True,
                        'xtol': 1e-4, 'maxiter': 2000, 'lambda_prior': 0, 'mu_prior': 0,
                        'alpha': .03, 'eta_ineq': 1., 'eta_eq': 1., 'eps': 1.,
                        'gp_std_threshold': 2, 'zero_prior': True,  # TODO

                        'simulation_dir': 'network',
                        'trajectory_bounds': None,
                        'use_opt_method': True,
                        'model_type': 'discrete',
                        'model_nonlinear_next_state': True,
                        'run_mpc': False,
                        'run_state_gp': False,
                        'run_cost_gp': False,
                        'compare_simulations': False,
                        'simulation_name': 'Known Models Offline N=3',
                        'simulation_comparisons': ['GP Offline N=5', 'GP Online N=5 maxiter=1'],
                        'imp_bounds_only': True,
                        'use_linear_test_values': True}

tcl_state_n_training_samples = [47, 95, 143, 191, 239, 287]
n_training_samples = [5, 10, 25, 50]
max_iters = [1, 5, 10, 25, 50]
n_horizons = [1, 3, 6, 12]
models = [('GP Models', 0.02), ('True Models', 0.025)]  # alpha=0.05, MaxIter=250, 1e-2 WORKS, trying with 0.03, 2000

simulations = [

    # 0 TCL State GP Simulations
    [{'simulation_name': f'GP Ntr={tcl_state_n_training_samples[s]}',
      'run_state_gp': True,
      'next_state_n_init_training_samples': tcl_state_n_training_samples[s],
      'next_state_max_n_training_samples': tcl_state_n_training_samples[s],
      'num_devices': {'tcl': 1, 'tcl2': 0, 'batt': 0, 'ip': 0}}
     for s in range(len(tcl_state_n_training_samples))],

    # 1 TCL Cost GP Simulations
    [{'simulation_name': f'GP Ntr={n_training_samples[s]}',
      'run_cost_gp': True,
      'stage_cost_n_init_training_samples': n_training_samples[s],
      'num_devices': {'tcl': 1, 'tcl2': 0, 'batt': 0, 'ip': 0},
      'stage_cost_max_n_training_samples': n_training_samples[s]}
     for s in range(len(n_training_samples))],

    # 2 IP State GP Simulations
    [{'simulation_name': f'GP Ntr={n_training_samples[s]}',
      'run_state_gp': True,
      'next_state_n_init_training_samples': n_training_samples[s],
      'next_state_max_n_training_samples': n_training_samples[s],
      'num_devices': {'tcl': 0, 'tcl2': 0, 'batt': 0, 'ip': 1}}
     for s in range(len(n_training_samples))],

    # 3 IP Cost GP Simulations
    [{'simulation_name': f'GP Ntr={n_training_samples[s]}',
      'run_cost_gp': True,
      'stage_cost_n_init_training_samples': n_training_samples[s],
      'num_devices': {'tcl': 0, 'tcl2': 0, 'batt': 0, 'ip': 1},
      'stage_cost_max_n_training_samples': n_training_samples[s]}
     for s in range(len(n_training_samples))],

    # 4 IP Offline Simulations todo 1) test with higher alpha - success 2) test with lower training range
    # todo 3) test w lower n training samples
    #  todo 4) try online w/ 100 iterations, lower step-size to avoid oscillations
    [{'simulation_name': f'IP {models[s][0]} Offline N=3',
      'maxiter': 1000, 'init_maxiter': 1000, 'n_horizon': 3, 'alpha': models[s][1], 'eta_eq': 1, 'eta_ineq': 1.,
      'eps': 1., 'run_mpc': True, 'n_simulation_steps': 300,
      'use_gp': models[s][0] == 'GP Models', 'mpc_t_step': 0.1, 'R': 0, 'num_devices': {'tcl': 0, 'batt': 0, 'ip': 1}}
     for s in range(len(models))],

    # # 5 IP Online Simulations
    [{'simulation_name': f'IP {models[s][0]} Online N=3 maxiter=100',
      'maxiter': 100, 'init_maxiter': 100, 'n_horizon': 3, 'alpha': models[s][1], 'eta_eq': 1., 'eta_ineq': 1.,
      'eps': 1., 'run_mpc': True, 'n_simulation_steps': 300,
      'use_gp': models[s][0] == 'GP Models',
      'mpc_t_step': 0.1, 'R': 0, 'num_devices': {'tcl': 0, 'tcl2': 0, 'batt': 0, 'ip': 1}} for s in range(len(models))],

    # 6 TCL GP Online Simulations for Different MaxIters
    [{'simulation_name': f'TCL GP Online N=6 maxiter={max_iters[s]}', 'run_mpc': True,
      'maxiter': max_iters[s], 'init_maxiter': max_iters[s], 'n_horizon': 6, 'alpha': 0.075, 'eta_ineq': 1, 'eta_eq': 1,
      'eps': 1.0, 'use_gp': True, 'mpc_t_step': 60 * 5, 'R': 0.1, 'num_devices': {'tcl': 5, 'tcl2': 0, 'batt': 0, 'ip': 0},
      'next_state_n_init_training_samples': 287,
      'next_state_max_n_training_samples': 287 + 144,
      'stage_cost_n_init_training_samples': 25,
      'stage_cost_max_n_training_samples': 50}
     for s in range(len(max_iters))],

    # 7 TCL GP Online Simulations for Different N
    [{'simulation_name': f'TCL GP Online N={n_horizons[s]} maxiter=100',
      'maxiter': 100, 'init_maxiter': 100, 'n_horizon': n_horizons[s], 'alpha': 0.075, 'eta_ineq': 1., 'eta_eq': 1.,
      # 0.1
      'eps': 1.0, 'run_mpc': True,
      'use_gp': True, 'mpc_t_step': 60 * 5, 'R': 0.1, 'num_devices': {'tcl': 5, 'tcl2': 0, 'batt': 0, 'ip': 0},
      'trajectory_bounds': None,
      'next_state_n_init_training_samples': 287,
      'next_state_max_n_training_samples': 287 + 144,
      'stage_cost_n_init_training_samples': 25,
      'stage_cost_max_n_training_samples': 50} for s in range(len(n_horizons))],

    # 8 TCL GP Function Offline Simulations
    [{'simulation_name': 'TCL GP Offline N=6', 'maxiter': 2000, 'init_maxiter': 2000, 'n_horizon': 6,
      'alpha': 0.075, 'eta_ineq': 1.,  # 0.5
      'eta_eq': 1., 'eps': 1.0, 'use_gp': True, 'mpc_t_step': 60 * 5, 'R': 0.1, 'run_mpc': True,
      'num_devices': {'tcl': 5, 'tcl2': 0, 'batt': 0, 'ip': 0},
      'next_state_n_init_training_samples': 287,
      'next_state_max_n_training_samples': 287 + 144,
      'stage_cost_n_init_training_samples': 25,
      'stage_cost_max_n_training_samples': 50}]
]
