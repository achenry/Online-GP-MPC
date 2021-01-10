import pandas as pd
import numpy as np

def exportTrainingData(n_batteries, n_tcls):
    # read original data into dataframes
    paths = ['./device_data/battery_data_full.csv', './device_data/tcl_data.csv']
    columns = [['TotalRealPower', 'availableenergy', 'maximumenergy', 'racksconnected'],
               ['T_outside',  # temperature outside building
                'T_zone_1',  # temperature
                'Q_solar_1',  # solar irradiance
                'Q_internal_1',  # internal irradiance
                'ms_dot_1',  # mass air flow
                'T_da',  # discharge air temperature
                'P_zone_1']]

    battery_data = pd.read_csv(paths[0], nrows=604800)
    n_samples = len(battery_data.index)
    power_noise = np.random.uniform(-0.05, 0.05, n_samples)
    soc_noise = np.random.uniform(-0.05, 0.05, n_samples)
    for d in range(n_batteries):
        new_battery_data = pd.DataFrame(columns=columns[0])
        battery_data = battery_data.loc[(battery_data['racksconnected'] == 83) |
                                        (battery_data['maximumenergy'] == 9408)]
        new_battery_data[f'P_{d + 1}'] = (battery_data['TotalRealPower'] * (1 + power_noise[d]))
        new_battery_data[f'soc_{d + 1}'] = ((battery_data['availableenergy'] / battery_data['maximumenergy'])
                                   * (1 + soc_noise[d]))
        new_battery_data[f'P_ref_{d + 1}'] = 0
        new_battery_data.to_csv(f'./device_data/battery_{d + 1}_data.csv', index=None)

    tcl_data = pd.read_csv(paths[1])
    for i in range(n_tcls):
        new_tcl_data = pd.DataFrame()
        for col in columns[1]:
            if col[-1] == '_1':
                new_tcl_data[col[:-2]] = tcl_data[col.replace('1', str(i + 1))]
            else:
                new_tcl_data[col] = tcl_data[col]

        new_tcl_data.to_csv(f'./device_data/tcl_{i + 1}_data.csv', index=None)


def exportDisturbanceData(n_batteries, n_tcls, battery_sampling_t_step):
    # read original data into dataframes
    paths = ['./device_data/tcl_data.csv', './device_data/battery_data_full.csv']
    columns = [['T_outside', 'T_da'], # temperature outside building, discharge air temperature
               ['P_ref']]

    tcl_data = pd.read_csv(paths[0])
    for d in range(n_tcls):
        new_tcl_data = pd.DataFrame()
        for col in columns[0]:
            new_tcl_data[col] = tcl_data[col]

        new_tcl_data.to_csv(f'./device_data/tcl_{d + 1}_disturbance_data.csv', index=None)

    six_hours = int(60 * 60 * 6 / battery_sampling_t_step)
    P_ref_six_hours = [1000 for i in range(six_hours)]
    Pneg_ref_six_hours = [-1000 for i in range(six_hours)]
    battery_disturbances = [P_ref_six_hours + Pneg_ref_six_hours for i in range(int(604800 / six_hours))]
    battery_disturbances =  battery_disturbances + [0 for i in range(604800 - len(battery_disturbances))]

    for d in range(n_batteries):
        new_battery_data = pd.DataFrame()
        for col in columns[1]:
            new_battery_data[col + f'_{d + 1}'] = battery_disturbances

        new_battery_data.to_csv(f'./device_data/battery_{d + 1}_disturbance_data.csv', index=None)

def main():
    exportTrainingData(5, 5)
    exportDisturbanceData(5, 5, 1)


if __name__ == '__main__':
    main()
