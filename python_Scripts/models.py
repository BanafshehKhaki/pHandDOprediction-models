import os
import re
import time
import joblib
import gc
from keras import backend as Kb
import numpy as np
import pandas as pd
from sklearn import tree
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from keras.models import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
import functions as func
from keras.utils import to_categorical
from numpy import argmax
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import fbeta_score
from sklearn.metrics import r2_score


def R2_measure(y_true, y_pred):
    return r2_score(y_true, y_pred)


def f2_measure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, labels=[1, 2],  beta=2, average='micro')


def main():

    # 'LR', 'DT', 'SVC', 'LSTM', 'NN', # 'MLP', 'CNN', 'LSTM', 'ConvLSTM', 'CNNLSTM', 'EncodeDecodeLSTMs'
    models = ['RF']
    targets = ['DOcategory', 'pHcategory', 'ph', 'dissolved_oxygen']
    sondefilename = 'leavon_wo_2019-07-01-2020-01-15'
    n_job = -1

    for model_name in models:
        print(model_name)

        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/bookOne/output_Cat_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'std_test_score': 'std_test_score', 'mean_test_score': 'mean_test_score', 'params': 'params', 'bestscore': 'bestscore', 'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta', 'imfeatures': 'imfeatures'}
            else:
                cat = 0
                directory = 'Results/bookOne/output_Reg_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'std_test_score': 'std_test_score', 'mean_test_score': 'mean_test_score', 'params': 'params', 'bestscore': 'bestscore', 'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mpe': 'mpe', 'rmse': 'rmse',  'R2': 'R2', 'imfeatures': 'imfeatures'}

            if not os.path.exists(directory):
                os.makedirs(directory)

            resultFileName = 'results_'+target+str(time.time())+'.csv'
            dfheader = pd.DataFrame(data=data, index=[0])
            dfheader.to_csv(directory+resultFileName,
                            index=False, header=False)

            path = 'Sondes_data/train/train_data/'
            method = 'OrgData'

            for n_steps in [1, 3, 6, 12]:  #
                for PrH_index in [1, 3, 6, 12, 24, 36, 48]:
                    files = [f for f in os.listdir(path) if f.endswith(
                        '.csv') and f.startswith(sondefilename)]
                    file = files[0]
                    print('Window: '+str(n_steps) + ' TH: ' +
                          str(PrH_index)+' '+method+' '+target)

                    dataset = pd.read_csv(path+file)

                    train_X_grid, train_y_grid, input_dim, features = func.preparedata(
                        dataset, PrH_index, n_steps, target, cat)
                    print(train_X_grid[0:1])

                    if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                        train_y_grid = to_categorical(train_y_grid, 3)
                    if model_name == 'LSTM' or model_name == 'NN':
                        n_job = 1

                    start_time = time.time()
                    model = func.algofind(model_name, input_dim, n_steps, cat)

                    if cat == 1:
                        metric = make_scorer(f2_measure)
                    else:
                        metric = make_scorer(R2_measure)

                    # cat_ix = train_X_grid[:, 7:]
                    # print(cat_ix[0:2])
                    # num_ix = train_X_grid[:, : 7]
                    # print(num_ix[0:2])
                    # one hot encode categorical, normalize numerical
                    # ct = ColumnTransformer(
                    #     [('c', OneHotEncoder(), cat_ix), ('n', StandardScaler(), num_ix)])

                    if model_name == 'RF' or model_name == 'DT':
                        pipeline = Pipeline(steps=[('model', model)])

                    else:  # model_name == 'LSTM' or model_name == 'NN':
                        pipeline = Pipeline(
                            steps=[('n', StandardScaler()), ('model', model)])

                    # else:
                    #     pipeline = Pipeline(
                    #         steps=[('transforms', ct), ('model', model)])

                    custom_cv = func.custom_cv_2folds(train_X_grid, 5)

                    if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                        gs = RandomizedSearchCV(
                            estimator=pipeline, param_distributions=func.param_grid['param_grid_'+model_name+str(cat)], n_iter=20, cv=custom_cv, verbose=0, random_state=42, n_jobs=n_job)
                        clf = gs.fit(train_X_grid, train_y_grid,
                                     model__class_weight={0: 1, 1: 50, 2: 100})
                    elif cat == 0 and (model_name == 'LSTM' or model_name == 'NN'):
                        gs = RandomizedSearchCV(
                            estimator=pipeline, param_distributions=func.param_grid['param_grid_'+model_name+str(cat)], n_iter=20, cv=custom_cv, verbose=0, random_state=42, n_jobs=n_job)
                        clf = gs.fit(train_X_grid, train_y_grid)
                    else:
                        gs = RandomizedSearchCV(
                            estimator=pipeline, param_distributions=func.param_grid['param_grid_'+model_name+str(cat)], n_iter=20, scoring=metric, cv=custom_cv, verbose=0, random_state=42, n_jobs=n_job)
                        clf = gs.fit(train_X_grid, train_y_grid)

                    test_Score = clf.cv_results_['mean_test_score'].mean()
                    test_std = clf.cv_results_['std_test_score'].mean()

                    print('Mean test scores: %.3f' % test_Score)

                    i = 1
                    custom_cv = func.custom_cv_2folds(train_X_grid, 3)
                    for train_index, test_index in custom_cv:
                        test_X = train_X_grid[test_index]
                        test_y = train_y_grid[test_index]

                        predictions = clf.predict(test_X)

                        fpath = 'predictions_' + method+target+'_Window' + \
                            str(n_steps) + '_TH' + \
                            str(PrH_index)+'_CV' + str(i)+file

                        if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                            test_y = argmax(test_y, axis=1)

                        cm0 = func.forecast_accuracy(predictions, test_y, cat)

                        plt.scatter(np.arange(len(test_y)),
                                    test_y, s=1)
                        plt.scatter(np.arange(len(predictions)),
                                    predictions, s=1)
                        plt.legend(['actual', 'predictions'],
                                   loc='upper right')

                        plt.savefig(directory+fpath+'.jpg')

                        plt.close()

                        data = {'Actual': test_y, 'Predictions': predictions}
                        print(test_y.shape)
                        print(predictions.shape)

                        df = pd.DataFrame(data=data)

                        df.to_csv(directory+fpath, index=False)

                        if cat == 1:
                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                    'file_names': fpath, 'std_test_score': [test_std], 'mean_test_score': [test_Score], 'params': [clf.best_params_], 'bestscore': [clf.best_score_], 'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]], 'imfeatures': [clf.best_estimator_]}
                        elif cat == 0:
                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                    'file_names': fpath, 'std_test_score': [test_std], 'mean_test_score': [test_Score], 'params': [clf.best_params_], 'bestscore': [clf.best_score_], 'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5], 'imfeatures': [clf.best_estimator_]}

                        df = pd.DataFrame(data=data, index=[0])
                        df.to_csv(directory+resultFileName,
                                  index=False, mode='a', header=False)

                        elapsed_time = time.time() - start_time
                        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                        i = i+1
                    # Kb.clear_session()
                    # gc.collect()
                    # del clf


if __name__ == "__main__":
    main()
