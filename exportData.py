import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
from Multipolyfit import multipolyfit as mpf

def exportTCLTrainingData(n_tcls):
    path = './device_data/tcl_data.csv'
    columns = ['T_outside',  # temperature outside building
                'T_zone_1',  # temperature,
                'P_zone_1']

    tcl_data = pd.read_csv(path) #.iloc[288:]
    # good_indices = (np.abs(stats.zscore(tcl_data[[f'T_zone_{d}' for d in range(1, n_tcls)]])) < 3).all(axis=1)
    # tcl_data = tcl_data.loc[good_indices]

    tcl_indices = [0, 2, 4]
    tcl_indices = [0, 1, 2, 3, 4]

    for i, idx in enumerate(tcl_indices):

        # if i == 3:
        #     input_i = 1
        # else:
        #     input_i = i

        input_i = idx
        output_i = i

        new_tcl_data = pd.DataFrame()
        new_cols = []
        for col in columns:

            if col[-2:] == '_1':
                input_col = col.replace('1', str(input_i + 1))
                output_col = col.replace('1', str(output_i + 1))
                new_tcl_data[output_col] = tcl_data[input_col]

                if col == 'P_zone_1':
                    new_tcl_data[output_col] = -new_tcl_data[output_col] * 12

                new_cols.append(output_col)

            else:
                new_tcl_data[col] = tcl_data[col]

        new_tcl_data = new_tcl_data.reset_index(drop=True)
        # diff = new_tcl_data[f'T_zone_{i + 1}'].diff()
        # new_tcl_data = new_tcl_data.loc[~(np.abs(diff - diff.mean()) > 3 * diff.std()).any()]

        new_tcl_data.to_csv(f'./device_data/tcl_{i + 1}_data.csv', index=None)

def exportBatteryTrainingData(n_batteries):
    # read original data into dataframes
    paths = ['./device_data/battery_data_full.csv']
    columns = [['TotalRealPower', 'availableenergy', 'maximumenergy', 'racksconnected']]

    battery_data = pd.read_csv(paths[0], nrows=3600 * 24 * 21)
    # clean data
    battery_data = battery_data.loc[(battery_data['racksconnected'] == 84) | (battery_data['maximumenergy'] == 9408)]
    E_max = 9408
    battery_data = battery_data.drop(columns=['racksconnected', 'maximumenergy', 'soctosdu', 'TotalRealPower_pmu'])

    battery_data['dt'] = pd.to_datetime(battery_data['dt'])
    battery_data.set_index('dt', inplace=True, drop=True)

    # battery_data = battery_data[(np.abs(stats.zscore(battery_data[['TotalRealPower']])) < 3).all(axis=1)]
    # battery_data.reset_index(drop=True, inplace=True)

    # dt_diff = (pd.to_datetime(battery_data['dt']).diff().dt.seconds.iloc[1:]) == 1
    # dt_diff.reset_index(drop=True, inplace=True)
    # battery_data = battery_data.iloc[:-1]
    #
    # battery_data['isgood'] = True
    # for idx, row in battery_data.iterrows():
    #     if dt_diff.iloc[idx] != 1:
    #         battery_data.iloc[idx]['isgood'] = False

    # battery_data = battery_data.iloc[::5 * 60]
    # battery_data.reset_index(drop=True, inplace=True)

    n_samples = len(battery_data.index)
    power_noise = np.zeros(n_samples)# np.random.uniform(-0.01, 0.01, n_samples)
    soc_noise = np.zeros(n_samples)# np.random.uniform(-0.01, 0.01, n_samples)
    for d in range(n_batteries):
        new_battery_data = pd.DataFrame()

        new_battery_data[f'dt'] = battery_data.index
        new_battery_data.set_index('dt', inplace=True)

        E_out = battery_data.resample('5T')['TotalRealPower'].sum() * (1 / 3600) * (1 + power_noise[d])
        new_battery_data[f'E_out_{d + 1}'] = E_out

        soc_sample = ((battery_data['availableenergy'] / E_max) * (1 + soc_noise[d])).resample('5Min')
        soc = soc_sample.apply(lambda arr: arr[0])
        new_battery_data[f'soc_{d + 1}'] = soc

        new_battery_data[f'P_ref_{d + 1}'] = 0

        new_battery_data.to_csv(f'./device_data/battery_{d + 1}_data.csv', index=None)

        soc_change = soc_sample.apply(lambda arr: arr[-1] - arr[0])

        # new_battery_data[f'dt_{d + 1}'] = (pd.to_datetime(battery_data['dt']).diff().dt.seconds / 3600).iloc[1:]
        # new_battery_data.drop(new_battery_data.index[-1], axis=0, inplace=True)

        # new_battery_data[f'P_out_{d + 1}'] = (battery_data['TotalRealPower'] * (1 + power_noise[d]))

        # new_battery_data[f'E_out_{d + 1}'] = new_battery_data[f'P_out_{d + 1}'] * new_battery_data[f'dt_{d + 1}']
        # new_battery_data[f'E_{d + 1}'] = battery_data['availableenergy'].iloc[:-1] * (1 + soc_noise[d])

        # new_battery_data['isgood'] = battery_data['isgood']

        # isgood_idx = new_battery_data['isgood'].index

        # E_out = new_battery_data[f'P_out_{d + 1}'].groupby(pd.TimeGrouper('5Min')).cumsum() * (1 / 3600)
        # P_out = new_battery_data[f'P_out_{d + 1}'].iloc[isgood_idx[:-1]]
        # soc = new_battery_data[f'soc_{d + 1}'].iloc[isgood_idx[:-1]]
        # soc_change = new_battery_data[f'soc_{d + 1}'].iloc[isgood_idx[:-1] + 1].reset_index(drop=True) \
        #              - new_battery_data[f'soc_{d + 1}'].iloc[isgood_idx[:-1]]

    dch_indices = (soc_change <= 0) & (E_out >= 0)
    ch_indices = (soc_change > 0) & (E_out < 0)
    bad_idx = (~dch_indices & ~ch_indices)

    soc = soc.loc[~bad_idx]
    soc_change = soc_change.loc[~bad_idx]
    E_out = E_out.loc[~bad_idx]
    dch_indices = dch_indices.loc[~bad_idx]
    ch_indices = ch_indices.loc[~bad_idx]

    # assumes discharging
    eta = (-E_out) / (soc_change * E_max)
    # eta_ch = (soc_change[ch_indices] * E_max) / (-E_out[ch_indices])
    # good_idx = (eta_ch <= 1) | (eta_dch <= 1)
    bad_idx = (dch_indices & (eta > 1)) | (ch_indices & (eta < 1))

    soc = soc.loc[~bad_idx]
    # soc_change = soc_change.loc[~bad_idx]
    E_out = E_out.loc[~bad_idx]
    eta = eta.loc[~bad_idx]
    # eta_dch = eta_dch.loc[eta_dch <= 1]
    # eta_ch = eta_ch.loc[eta_ch <= 1]

    # good_idx = ((0 <= eta_dch) & (eta_dch <= 1) & (0 <= eta_ch) & (eta_ch <= 1)).index
    # soc = soc.iloc[good_idx]
    x_train = np.hstack([soc.values[:-1, np.newaxis], E_out.values[:-1, np.newaxis]])
    y_train = soc.values[1:]
    model, coeffs, power = mpf(x_train, y_train, 2, model_out=True)
    x_test = x_train # np.vstack([np.linspace(0, 1, 100), np.linspace(-9000, 9000, 100)]).T
    # y_pred = np.polyval(coeffs, x_test)
    y_pred = np.array([model(x, y) for x, y in zip(x_test[:, 0], x_test[:, 1])])
    y_true = y_train

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plot training data
    ax.scatter(x_train[:, 0], x_train[:, 1], y_train, label='Training Data', color='tab:blue', s=2.)

    # plot gp prediction
    ax.plot_trisurf(x_test[:, 0], x_test[:, 1], y_pred,
                    color='tab:red',
                    label='GP Posterior Mean', antialiased=True, alpha=0.5, linewidth=0.5)
    fig.show()

    fig, ax = plt.subplots(1, 1)
    # eta_ch_plot = ((y_true[x_test[:, 1] < 0] - x_test[:, 0][x_test[:, 1] < 0]) * E_max) \
    #               / (-x_test[:, 1][x_test[:, 1] < 0])
    # eta_dch_plot = (-x_test[:, 1][x_test[:, 1] >= 0]) \
    #                / ((y_true[x_test[:, 1] >= 0] - x_test[:, 0][x_test[:, 1] >= 0]) * E_max)
    ax.scatter(E_out.loc[eta > 1], 1 / eta.loc[eta > 1])
    ax.scatter(E_out.loc[eta <= 1], eta.loc[eta <= 1])
    ax.set_xlabel('E_out')
    ax.set_ylabel('eta')
    fig.show()

    for input_dim, label in zip([0, 1], ['soc', 'E_out']):
        fig, ax = plt.subplots(1, 1, frameon=False)
        fig.suptitle('soc')

        # plot based on first dim of y la only
        ax.set_xlabel(label)

        X_train = x_train[:, input_dim]
        X_test = x_test[:, input_dim]

        sort_idx = np.argsort(X_test, axis=0)
        y_pred = y_pred[sort_idx]
        X_test = X_test[sort_idx]
        y_true = y_true[sort_idx]

        ax.scatter(X_test, y_true, color='tab:blue', label='True Function')

        # plot gp prediction and variance
        ax.scatter(X_test, y_pred, color='tab:red', label='GP Posterior Mean')
        # plot training data
        ax.scatter(X_train, y_train, color='tab:blue', label='Training Data')

        ax.legend(bbox_to_anchor=(1, 1))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.show()

    # E_max = 9408
    # P_out = new_battery_data[f'P_out_{d + 1}']
    # E_out = P_out * 5 * 60 / 3600
    # bad_data_indices = E_out < -600
    # E_out = E_out[~bad_data_indices].values[:-1]
    # soc_change = new_battery_data[f'soc_{d + 1}'][~bad_data_indices].diff()[1:]
    # dch_indices = soc_change >= 0
    # ch_indices = soc_change < 0
    #
    # eta_dch = E_out[dch_indices] / (soc_change[dch_indices] * E_max)
    # eta_ch = (soc_change[ch_indices] * E_max) / E_out[ch_indices]
    #
    # ax[0].scatter(P_out[ch_indices][(eta_ch <= 1) & (eta_ch >= 0)], 100 * eta_ch[(eta_ch <= 1) & (eta_ch >= 0)])
    # ax[0].set_title('Charging Efficiency vs Discharging Energy')
    # ax[0].set_xlabel('kWh')
    # ax[0].set_ylabel('%')
    # ax[1].scatter(P_out[dch_indices][(eta_dch <= 1) & (eta_dch >= 0)], 100 * eta_dch[(eta_dch <= 1) & (eta_dch >= 0)])
    # ax[1].set_title('Discharging Efficiency vs Discharging Energy')
    # ax[1].set_xlabel('kWh')
    # ax[1].set_ylabel('%')
    # fig.show()


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
    exportTCLTrainingData(5)
    # exportBatteryTrainingData(1)
    # exportDisturbanceData(1, 1, 1)


if __name__ == '__main__':
    main()
