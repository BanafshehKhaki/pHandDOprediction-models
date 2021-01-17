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


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def main():

    models = ['RF']  # 'LSTM', 'NN', 'LR', 'RF', 'DT', 'SVC',
    targets = ['ph']  # ['DOcategory', 'pHcategory'] # 'ph','dissolved_oxygen'
    # ph TH: 24,36,48
    sondefilename = 'leavon_wo_2019-07-01-2020-01-15'
    n_job = -1

    for model_name in models:
        print(model_name)

        for target in targets:
            if target.find('category') > 0:
                cat = 1
                directory = 'Results/balance_data/output_Cat_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'std_test_score': 'std_test_score', 'mean_test_score': 'mean_test_score', 'params': 'params', 'bestscore': 'bestscore', 'F1_0': 'F1_0', 'F1_1': 'F1_1', 'P_0': 'P_0', 'P_1': 'P_1', 'R_0': 'R_0', 'R_1': 'R_1', 'acc0_1': 'acc0_1', 'F1_0_1': 'F1_0_1', 'F1_all': 'F1_all', 'fbeta': 'fbeta', 'imfeatures': 'imfeatures', 'best_thresh_0': 'best_thresh_0', 'best_thresh_1': 'best_thresh_1', 'best_thresh_2': 'best_thresh_2'}
            else:
                cat = 0
                directory = 'Results/balance_data/output_Reg_' + \
                    model_name+'/oversampling_cv_models/'
                data = {'target_names': 'target_names', 'method_names': 'method_names', 'window_nuggets': 'window_nuggets', 'temporalhorizons': 'temporalhorizons', 'CV': 'CV',
                        'file_names': 'file_names',  'std_test_score': 'std_test_score', 'mean_test_score': 'mean_test_score', 'params': 'params', 'bestscore': 'bestscore', 'mape': 'mape', 'me': 'me', 'mae': 'mae', 'mpe': 'mpe', 'rmse': 'rmse', 'R2': 'R2', 'imfeatures': 'imfeatures'}

            if not os.path.exists(directory):
                os.makedirs(directory)

            resultFileName = 'results_'+target+str(time.time())+'.csv'
            dfheader = pd.DataFrame(data=data, index=[0])
            dfheader.to_csv(directory+resultFileName,
                            index=False, header=False)

            if model_name == 'DT' or model_name == 'RF':
                path = 'Sondes_data/train/train_data/'
                method = 'OrgData'
            else:
                method = 'StandardScaler'
                path = 'Sondes_data/train/train_data_normalized/'+method+'/'+target+'/'

            for n_steps in [1, 3, 6, 12]:
                for PrH_index in [1, 3, 6, 12, 24, 36, 48]:
                    files = [f for f in os.listdir(path) if f.endswith(
                        '.csv') and f.startswith(sondefilename)]
                    file = files[0]
                    print('Window: '+str(n_steps) + ' TH: ' +
                          str(PrH_index)+' '+method+' '+target)

                    dataset = pd.read_csv(path+file)
                    train_X_grid, train_y_grid, input_dim, features = func.preparedata(
                        dataset, PrH_index, n_steps, target, cat)

                    if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                        train_y_grid = to_categorical(train_y_grid, 3)
                    if model_name == 'LSTM' or model_name == 'NN':
                        n_job = 1

                    start_time = time.time()

                    # resample = SMOTETomek(tomek=TomekLinks(
                    #     sampling_strategy='majority'))
                    # print(train_y_grid[train_y_grid.argmax(axis=1)==2])

                    model = func.algofind(model_name, input_dim, n_steps, cat)
                    # ('r', resample),
                    # if cat == 1:
                    #     model = CalibratedClassifierCV(
                    #         model, method='isotonic')

                    pipeline = Pipeline(steps=[('model', model)])

                    custom_cv = func.custom_cv_2folds(train_X_grid, 5)
                    gs = RandomizedSearchCV(
                        estimator=pipeline, param_distributions=func.param_grid['param_grid_'+model_name+str(cat)], n_iter=10, cv=custom_cv, verbose=0, random_state=42, n_jobs=n_job)

                    if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                        clf = gs.fit(train_X_grid, train_y_grid,
                                     model__class_weight={0: 1, 1: 50, 2: 100})
                    else:
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
                        # predict_mine = []
                        fpath = 'predictions_' + method+target+'_Window' + \
                            str(n_steps) + '_TH' + \
                            str(PrH_index)+'_CV' + str(i)+file

                        if cat == 1:
                            # predict probabilities
                            yhat = clf.predict_proba(test_X)
                            # print(yhat[100:103])
                            y = label_binarize(test_y, classes=[0, 1, 2])
                            # print(y[100:103])

                            # roc_curve
                            fpr = dict()
                            tpr = dict()
                            roc_auc = dict()
                            best_thresh = dict()
                            for i in range(3):
                                fpr[i], tpr[i], thresholds = roc_curve(
                                    y[:, i], yhat[:, i])
                                roc_auc[i] = auc(fpr[i], tpr[i])
                                J = tpr[i] - fpr[i]
                                # get the best threshold
                                ix = argmax(J)
                                best_thresh[i] = thresholds[ix]
                                print('Best Threshold=%f, roc_auc=%.3f' %
                                      (best_thresh[i], roc_auc[i]))

                            # Compute micro-average ROC curve and ROC area
                            fpr["micro"], tpr["micro"], _ = roc_curve(
                                y.ravel(), yhat.ravel())
                            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                            plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(
                                roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

                            colors = cycle(
                                ['aqua', 'darkorange', 'cornflowerblue'])
                            for i, color in zip(range(3), colors):
                                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                                         label='ROC curve of class {0} (area = {1:0.2f})'
                                         ''.format(i, roc_auc[i]))
                            # plot the roc curve for the model
                            plt.plot([0, 1], [0, 1], linestyle='--',
                                     label='No Skill')
                            # axis labels
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title(
                                'Some extension of Receiver operating characteristic to multi-class')
                            plt.legend(loc="lower right")
                            # show the plot
                            plt.savefig(directory+fpath+'ROC_curve.jpg')
                            plt.close()

                        if cat == 1 and (model_name == 'LSTM' or model_name == 'NN'):
                            test_y = argmax(test_y, axis=1)
                            # predictions = argmax(predictions, axis=1)
                        if cat == 0:
                            predictions, test_y = func.transform(
                                predictions, test_y, method, target, file)

                        cm0 = func.forecast_accuracy(predictions, test_y, cat)

                        plt.scatter(np.arange(len(test_y)),
                                    test_y, s=1)
                        plt.scatter(np.arange(len(predictions)),
                                    predictions, s=1)
                        plt.legend(['actual', 'predictions'],
                                   loc='upper right')

                        plt.savefig(directory+fpath+'.jpg')

                        plt.close()

                        # data = {'Actual': test_y, 'Predictions': predictions}
                        print(test_y.shape)
                        print(predictions.shape)

                        # if model_name == 'RF':
                        #     df = pd.DataFrame(data=data)
                        # else:
                        #     df = pd.DataFrame(data=data, index=[0])
                        # df.to_csv(directory+fpath, index=False)

                        if cat == 1:
                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                    'file_names': fpath, 'std_test_score': [test_std], 'mean_test_score': [test_Score], 'params': [clf.best_params_], 'bestscore': [clf.best_score_], 'F1_0': cm0[0], 'F1_1': cm0[1], 'P_0': cm0[2], 'P_1': cm0[3], 'R_0': cm0[4], 'R_1': cm0[5], 'acc0_1': cm0[6], 'F1_0_1': cm0[7], 'F1_all': cm0[8], 'fbeta': [cm0[9]], 'imfeatures': [clf.best_estimator_], 'best_thresh_0': best_thresh[0], 'best_thresh_1': best_thresh[1], 'best_thresh_2': best_thresh[2]}
                        elif cat == 0:
                            data = {'target_names': target, 'method_names': method, 'window_nuggets': n_steps, 'temporalhorizons': PrH_index, 'CV': i,
                                    'file_names': fpath, 'std_test_score': [test_std], 'mean_test_score': [test_Score], 'params': [clf.best_params_], 'bestscore': [clf.best_score_], 'mape': cm0[0], 'me': cm0[1], 'mae': cm0[2], 'mpe': cm0[3], 'rmse': cm0[4], 'R2': cm0[5], 'imfeatures': [clf.best_estimator_]}

                        df = pd.DataFrame(data=data, index=[0])
                        df.to_csv(directory+resultFileName,
                                  index=False, mode='a', header=False)

                        elapsed_time = time.time() - start_time
                        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                        i = i+1
                    Kb.clear_session()
                    gc.collect()
                    del clf


if __name__ == "__main__":
    main()
