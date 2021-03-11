# generate latex tables and copy figures into exports directory

import pandas as pd
import os
import numpy as np
import shutil
import re


class GenerateExports:

    def __init__(self, export_dir, results_dir):
        self.export_dir = export_dir
        self.results_dir = results_dir

        # else:
        #     for root, _, files in os.walk(self.export_dir):
        #         for file in files:
        #             os.remove(os.path.join(f'{root}', file))

    def generate_mpc_params_table(self):

        if os.path.exists(f'{self.results_dir}/mpc_results.csv'):
            mpc_results_df = pd.read_csv(f'{self.results_dir}/mpc_results.csv', engine='python', index_col=0)

            # generate latex table of mpc simulation parameters
            table_dir = f'{self.export_dir}/mpc_parameters.txt'

            table_file = open(f'{table_dir}', 'a')

            for i, row in mpc_results_df.iterrows():

                use_gp = row['use_gp']

                function_type = "GP Models" if use_gp else "True Models"

                table_row = f'{function_type} & ' \
                            f'{row["maxiter"]} & ' \
                            f'{row["n_simulation_steps"]} & ' \
                            f'{row["n_horizon"]} & ' \
                            f'{row["alpha"]} & ' \
                            f'{row["eta_ineq"]} & ' \
                            f'{row["eta_eq"]} & ' \
                            f'{row["eps"]} \\\\\n'

                table_file.write(table_row)

            table_file.close()

    def generate_gp_params_table(self):

        if os.path.exists(f'{self.results_dir}/gp_results.csv'):

            gp_results_df = pd.read_csv(f'{self.results_dir}/gp_results.csv', engine='python', index_col=0)

            # generate latex table of mpc simulation parameters
            test_types = ['stage_cost', 'stage_cost', 'state_var', 'state_var']
            test_keywords = ['Stage Cost (TCL', 'Stage Cost (IP',
                             'State Variation (TCL', 'State Variation (IP']
            table_dirs = [f'{self.export_dir}/{test_keywords[t]}_parameters.txt'
                          for t in range(len(test_keywords))]
            test_numbers = range(len(test_types))

            for t in test_numbers:

                table_file = open(f'{table_dirs[t]}', 'w')

                for i, row in gp_results_df.iterrows():

                    if test_keywords[t] in row['Prediction Name']:

                        length_scales = np.asarray([g[0] for g in re.findall('(-*\d*\.\d*(e-*\d+)*)+',
                                                                             row['Length Scale'])]).astype(float)

                        table_row = f'{row["No. Training Samples"]} & ' \
                                    f'{[np.round(x, 4) for x in length_scales]} & ' \
                                    f'{row["Output Variance"]} & ' \
                                    f'{row["Measurement Noise"]} & ' \
                                    f'{np.round(row["Score"], 4)}\\\\\n'

                        table_file.write(table_row)

                table_file.close()

    def copy_figures(self):

        keywords = ['_gp_', 'traj_plot.png', 'var_plot.png', 'conv_plot.png', 'error_plot.png', 'gp_score.png'] # 'conv_plot.png'

        for root, dirs, files in os.walk(f'{self.results_dir}'):
            for file in files:
                if any([keyword in file for keyword in keywords]):
                    shutil.copyfile(f'{root}/{file}', f'{self.export_dir}/{os.path.basename(root)}_{file}')