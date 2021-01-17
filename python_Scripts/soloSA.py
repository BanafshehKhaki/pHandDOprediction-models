from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import numpy as np

# https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html
# https://salib.readthedocs.io/en/latest/basics.html#what-is-sensitivity-analysis


def custom_cv_kfolds_testdataonly(X, kfolds):
    n = X.shape[0]
    # print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-6, step=6, dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        # print(idx[0:10])
        yield idx[:int(len(idx))]
        i = i+1


def split_sequences(data, n_steps, n_step_out):
    data = data.values
    X, y = list(), list()

    for i in range(len(data)):
        end_ix = i + n_steps*6
        if end_ix > len(data):
            break

        Kx = np.empty((1, 12))
        for index in np.arange(i, i+(n_steps*6), step=6, dtype=int):
            eachhour = index + 6
            if eachhour > len(data) or i+(n_steps*6) > len(data):
                break

            a = data[index: eachhour, : (-1*n_step_out)]
            hourlymean_x = np.round(np.mean(a, axis=0), decimals=2)
            hourlymean_y = data[eachhour-1, (-1*n_step_out):]

            hourlymean_x = hourlymean_x.reshape((1, hourlymean_x.shape[0]))
            if index != i:
                Kx = np.append(Kx, hourlymean_x, axis=0)
            else:
                Kx = hourlymean_x

        X.append(Kx)
        y.append(hourlymean_y)
    # print(np.array(X).shape)
    return np.array(X), np.array(y)


def temporal_horizon(df, pd_steps, target):
    for pd_steps in [1, 3, 6, 12, 24, 36, 48, 60, 72]:
        pd_steps = pd_steps * 6
        target_values = df[[target]]
        target_values = target_values.drop(
            target_values.index[0: pd_steps], axis=0)
        target_values.index = np.arange(0, len(target_values[target]))
        df['Target_'+target+'_t'+str(pd_steps)] = target_values

    df = df.drop(df.index[len(df.index)-(72*6): len(df.index)], axis=0)
    return df


path = 'Sondes_data/test_Summer/'
testpath = 'Sondes_data/test_Summer/'
method = 'OrgData'
n_steps_in = 24

for PrH_index in [12]:

    dataset = pd.read_csv(path+'leavon.csv')
    dataset = dataset[['Water_Temperature_at_Surface', 'ysi_chlorophyll',
                       'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph',  'year', 'month', 'day', 'hour']]
    dataset_bgsusd = pd.read_csv(path+'bgsusd_all.csv')

    dataset_bgsusd = dataset_bgsusd[['Water_Temperature_at_Surface', 'ysi_chlorophyll',
                                     'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph', 'year', 'month', 'day', 'hour']]

    dataset = temporal_horizon(
        dataset, PrH_index, 'dissolved_oxygen')

    dataset_bgsusd = temporal_horizon(
        dataset_bgsusd, PrH_index, 'dissolved_oxygen')

    n_steps_out = 9
    train_X_grid, y = split_sequences(
        dataset, n_steps_in, n_steps_out)
    print(train_X_grid.shape)

    n_features = train_X_grid.shape[2]
    print('n_fetures: ' + str(n_features))

    train_X_grid_bgsusd, train_y_grid_bgsusd = split_sequences(
        dataset_bgsusd, n_steps_in, n_steps_out)

    train_X_grid = train_X_grid.reshape(
        train_X_grid.shape[0], train_X_grid.shape[1]*train_X_grid.shape[2])
    print(train_X_grid.shape)

    train_X_grid_bgsusd = train_X_grid_bgsusd.reshape(
        train_X_grid_bgsusd.shape[0], train_X_grid_bgsusd.shape[1]*train_X_grid_bgsusd.shape[2])

    XX = np.hstack((train_X_grid_bgsusd, train_X_grid))
    input_dim = XX.shape
    print(input_dim)

custom_cv = custom_cv_kfolds_testdataonly(XX, 1)
for test_index in custom_cv:
    test_X = XX[test_index]
    test_y = y[test_index]
    print(test_X.shape)

    maxvals = test_X.max(axis=0)
    minvals = test_X.min(axis=0)

    problem = {
        'num_vars': test_X.shape[1],
        'names': [i for i in range(0, test_X.shape[1])],
        'bounds': [[pair[0], pair[1]] for pair in zip(minvals, maxvals)]
    }
# print(problem)
# 'Water_Temperature_at_Surface', 'ysi_chlorophyll', 'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph',  'year', 'month', 'day', 'hour'
    results = pd.read_csv('Results/bookThree/2sondes/output_Reg_LSTM/final_models/Results_test_othersondes_with_their_own_trained_model/' +
                          'predictions_OrgDatadissolved_oxygen_Window24_CV50leavon.csv')
    results = results[0:3464]
    Y = results['p+12'].values
    print(Y.shape)
    Si = sobol.analyze(problem, Y)
    print(Si.keys)
    S1 = Si['S1']
    S1 = S1.reshape(48, 9)
# # Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf". The _conf keys store the corresponding confidence intervals, typically with a confidence level of 95%. Use the keyword argument print_to_console=True to print all indices. Or, we can print the individual values from Si as shown below.
    df = pd.DataFrame(S1, columns=['Water_Temperature_at_Surface', 'ysi_chlorophyll',
                                   'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph', 'year', 'month', 'day', 'hour'])
    directory = 'Results/bookThree/2sondes/output_Reg_LSTM'+'/final_models/'
    df.to_csv(directory+'sobol_SA_S1_pred_DO_TH12.csv', index=False)

    ST = Si['ST']
    ST = ST.reshape(48, 9)
    df = pd.DataFrame(ST, columns=['Water_Temperature_at_Surface', 'ysi_chlorophyll',
                                   'dissolved_oxygen_saturation', 'dissolved_oxygen', 'ph', 'year', 'month', 'day', 'hour'])
    directory = 'Results/bookThree/2sondes/output_Reg_LSTM'+'/final_models/'
    df.to_csv(directory+'sobol_SA_ST_pred_DO_TH12.csv', index=False)

    print(Si['S2'].shape)
    S2 = Si['S2']
    df = pd.DataFrame(S2)
    directory = 'Results/bookThree/2sondes/output_Reg_LSTM'+'/final_models/'
    df.to_csv(directory+'sobol_SA_S2_pred_DO_TH12.csv', index=False)
