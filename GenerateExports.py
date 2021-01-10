# generate latex tables and copy figures into exports directory

import pandas as pd
import os
import numpy as np
import shutil
import re


class GenerateExports:

    def __init__(self):
        self.export_dir = './results/exports'

        if not os.path.exists('./results/exports'):
            os.mkdir('./results/exports')
        else:
            for root, _, files in os.walk(self.export_dir):
                for file in files:
                    os.remove(os.path.join(f'{root}', file))

    def generate_mpc_params_table(self):

        if os.path.exists('./results/mpc_results.csv'):
            mpc_results_df = pd.read_csv('./results/mpc_results.csv', engine='python', index_col=0)

            # generate latex table of mpc simulation parameters
            test_types = ['offline_mpc', 'online_mpc']
            test_keywords = ['Offline', 'Online']
            table_dirs = [f'{self.export_dir}/{test_type}_parameters.txt' for test_type in test_types]
            test_numbers = [1 for test_type in test_types]

            for t in range(len(test_types)):

                table_file = open(f'{table_dirs[t]}', 'a')

                for i, row in mpc_results_df.iterrows():

                    if any([keyword in row['simulation_name'] for keyword in test_keywords]):

                        use_next_state_gp_approx = \
                            [True if elem == 'True' else False for elem in
                             re.findall('(\w*)', row["use_next_state_gp_approx"])]

                        use_cost_gp_approx = True if row["use_cost_gp_approx"] == 'True' else False

                        function_type = "GP Approximation" if any(use_next_state_gp_approx + [use_cost_gp_approx]) \
                            else "Known Models"

                        table_row = f'{test_numbers[t]} & ' \
                                    f'{function_type} & ' \
                                    f'{row["maxiter"]} & ' \
                                    f'{row["n_simulation_steps"]} & ' \
                                    f'{row["n_horizon"]} & ' \
                                    f'{row["alpha"]} & ' \
                                    f'{row["eta"]} & ' \
                                    f'{row["eps"]} \\\\\n'

                        table_file.write(table_row)
                        test_numbers[t] += 1

                table_file.close()

    def generate_gp_params_table(self):

        if os.path.exists('./results/gp_results.csv'):

            gp_results_df = pd.read_csv('./results/gp_results.csv', engine='python', index_col=0)

            # generate latex table of mpc simulation parameters
            test_types = ['next_state_error_1_gp', 'cost_gp']
            test_keywords = ['GP Ntr', 'GP Ntr']
            table_dirs = [f'{self.export_dir}/{test_type}_parameters.txt' for test_type in test_types]
            test_numbers = [1 for test_type in test_types]

            for t in range(len(test_types)):

                table_file = open(f'{table_dirs[t]}', 'a')

                for i, row in gp_results_df.iterrows():

                    if any([keyword in row['Prediction Name'] for keyword in test_keywords]):

                        table_row = f'{test_numbers[t]} & ' \
                                    f'{row["No. Training Samples"]} & ' \
                                    f'{np.round(row["Length Scale"], 4)} & ' \
                                    f'{row["Output Variance"]} & ' \
                                    f'{row["Measurement Noise"]} & ' \
                                    f'{np.round(row["Score"], 4)}\\\\\n'

                        table_file.write(table_row)
                        test_numbers[t] += 1

                table_file.close()

    def copy_figures(self):

        keywords = ['gp.png', 'traj_plot.png', 'var_plot.png', 'gp_score.png'] # 'conv_plot.png'

        for root, dirs, files in os.walk('./results'):
            for file in files:
                if any([keyword in file for keyword in keywords]):
                    shutil.copyfile(f'{root}/{file}', f'{self.export_dir}/{os.path.basename(root)}_{file}')