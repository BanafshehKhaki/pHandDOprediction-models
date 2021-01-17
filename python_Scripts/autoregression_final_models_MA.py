import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import re
import datetime
import sklearn.metrics as skm
import functions as func
import os
import warnings
warnings.filterwarnings("ignore")


def temporal_horizon(df, pd_steps, target):
    pd_steps = pd_steps * 6
    target_values = df[[target]]
    target_values = target_values.drop(
        target_values.index[0: pd_steps], axis=0)
    target_values.index = np.arange(0, len(target_values[target]))

    df = df.drop(
        df.index[len(df.index)-pd_steps: len(df.index)], axis=0)
    df['Target_'+target] = target_values
    print('Target_'+target)
    return df


def ARIMAregression(train, test, p, d, q):
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    r2_score = skm.r2_score(test, predictions)
    print('Test r2_score: %.3f' % r2_score)
    # plot forecasts against actual outcomes
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()
    return predictions


def movingAverage(train_X, train_y):

    # calculate residuals for # persistence model on training set
    train_resid = [train_y[i]-train_X[i] for i in range(len(train_X))]

    # model the training set residuals
    model = AR(train_resid)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    lag = train_resid[len(train_resid)-window:]
    return coef, lag


def AutoRegression(train):

    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    lag = train[-window:]
    return coef, lag


def custom_cv_2folds(X, kfolds, th):
    n = X.shape[0]
    print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-(th*6), step=(th*6), dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        print(idx[0: 10])
        yield idx[: int(len(idx)*0.7)], idx[int(len(idx)*0.7):]
        i = i+1


def custom_cv_kfolds_testdataonly(X, kfolds, th):
    n = X.shape[0]
    # print(n)
    print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-(th*6), step=(th*6), dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        print(idx[0:10])
        yield idx[:int(len(idx))]
        i = i+1


def predict(coef, lag, window):
    yhat = coef[0]
    for d in range(1, window):
        yhat += coef[d] * lag[-d]
    return yhat


def main():

    models = ['MA']
    targets = ['ph', 'dissolved_oxygen']  # 'pHcategory', 'DOcategory'
    sondefilename = 'leavon'

    for model_name in models:
        print(model_name)

        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookThree/1sonde/output_Cat_' + \
                    model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names', 'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta'}
            else:
                cat = 0
                directory = 'Results/bookThree/1sonde/output_Reg_' + \
                    model_name+'/final_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mpe': 'mpe', 'rmse': 'rmse',  'R2': 'R2'}

            if not os.path.exists(directory):
                os.makedirs(directory)

            directoryresult = directory + 'Results/'
            if not os.path.exists(directoryresult):
                os.makedirs(directoryresult)
            print(directoryresult)
            testsondefilename = 'utlcp'
            resultFileName = 'results_'+testsondefilename + '_' + \
                target+str(time.time())+'.csv'
            dfheader = pd.DataFrame(data=data, index=[0])
            dfheader.to_csv(directoryresult+resultFileName,
                            index=False, header=False)

            path = 'Sondes_data/train_Summer/'
            testpath = 'Sondes_data/test_Summer/'
            method = 'OrgData'

            for n_steps in [1]:
                for PrH_index in [1, 3, 6, 12, 24, 36, 48, 60, 72]:  # 1, 3, 6, 12,
                    # files = [f for f in os.listdir(path) if f.endswith(
                    #     '.csv') and f.startswith(sondefilename)]
                    # file = files[0]
                    # print('Window: '+str(n_steps) + ' TH: ' +
                    #       str(PrH_index)+' '+method+' '+target)

                    # dataset = pd.read_csv(path+file)

                    # ######################
                    # # FOR MA
                    # ######################
                    # dataset = temporal_horizon(dataset, PrH_index, target)
                    # train = dataset[target]
                    # train_target = dataset['Target_'+target]
                    # print(train.head())
                    # print(train_target.head())

                    # custom_cv = func.custom_cv_kfolds_testdataonly(
                    #     train, 1)

                    # for train_index in custom_cv:
                    #     train = train[train_index].values
                    #     train_target = train_target[train_index].values

                        # coef, lag = movingAverage(
                        #     train, train_target)
                        # np.save(directory+'MA_model_'+target +
                        #                 '_'+str(PrH_index)+'.npy')
                        # np.save(directory+'MA_data_'+target +
                        #                 '_'+str(PrH_index)+'.npy', lag)

                    coef = np.load(directory+'MA_model_'+target +
                                   '_'+str(PrH_index)+'.npy')
                    lag = np.load(directory+'MA_data_'+target +
                                  '_'+str(PrH_index)+'.npy')

                    ######################
                    # TEST sets
                    ######################
                    # start_time = time.time()
                    # testsondefilename = re.sub('wo_', '', sondefilename)

                    files = [f for f in os.listdir(testpath) if f.endswith(
                        '.csv')and f.startswith(testsondefilename)]
                    file1 = files[0]
                    print('Window: ' + str(len(lag)) + ' TH: ' +
                          str(PrH_index)+' '+method+' '+target+file1)

                    testdataset = pd.read_csv(testpath+file1)
                    testdataset = temporal_horizon(
                        testdataset, PrH_index, target)

                    test = testdataset[target]
                    test_target = testdataset['Target_'+target]
                    # print(test.head())
                    # print(test_target.head())

                    i = 1
                    custom_cv = func.custom_cv_kfolds_testdataonly(
                        test, 100)
                    for test_index in custom_cv:
                        test_y = test[test_index].values
                        # for MA
                        test_y_targets = test_target[test_index].values

                        # walk forward over time steps in test
                        history = [lag[i] for i in range(len(lag))]
                        predictions = list()
                        for t in range(len(test_y)):
                            # persistence
                            yhat = test_y[t]
                            # predict error
                            length = len(history)
                            window = len(coef)
                            hl = [history[i]
                                  for i in range(length-window, length)]
                            pred_error = predict(coef, hl, window)
                            yhat = yhat + pred_error
                            predictions.append(yhat)
                            error = test_y_targets[t] - yhat
                            history.append(error)

                        if cat == 1:
                            predictions = np.array(
                                predictions).astype(int)

                        fpath = 'predictions_' + method+target+'_Window' + \
                            str(n_steps) + '_TH' + \
                            str(PrH_index)+'_CV' + \
                            str(i) + testsondefilename
                        # '_vals_'+str(p)+'_'+str(d) + \
                        # '_'+str(q)+'_'+\
                        # print(len(predictions))
                        # print(len(test_y_targets))
                        cm0 = func.forecast_accuracy(
                            predictions, test_y_targets, cat)

                        if i % 10 == 0:
                            plt.scatter(np.arange(len(test_y_targets)),
                                        test_y, s=1)
                            plt.scatter(np.arange(len(predictions)),
                                        predictions, s=1)
                            plt.legend(['actual', 'predictions'],
                                       loc='upper right')
                            plt.savefig(directoryresult+fpath+'.png')
                            plt.close()

                            data = {'Actual': test_y_targets,
                                    'Predictions': predictions}
                            df = pd.DataFrame(data=data)
                            df.to_csv(directoryresult+fpath, index=False)

                        if cat == 1:
                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                    'file_names': testsondefilename, 'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]]}
                        elif cat == 0:
                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                    'file_names': testsondefilename,  'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5]}

                        df = pd.DataFrame(data=data, index=[0])
                        df.to_csv(directoryresult+resultFileName,
                                  index=False, mode='a', header=False)

                        i = i+1


if __name__ == "__main__":
    main()
