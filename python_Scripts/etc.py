# calculate roc curves
precision = dict()
recall = dict()
fscore = dict()
average_precision = dict()
 best_thresh = dict()
  colors = ['aqua', 'darkorange', 'cornflowerblue']
   for i in range(3):
        precision[i], recall[i], thresholds = precision_recall_curve(
            y[:, i], yhat[:, i])
        # convert to f-measure
        average_precision[i] = average_precision_score(y[:, i], yhat[:, i])
        fscore[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
        # locate the index of the largest f-measure
        ix = argmax(fscore[i])
        best_thresh[i] = thresholds[ix]
        print('Best Threshold=%f, F-measure=%.3f' %
              (thresholds[ix], fscore[i][ix]))
        # plot the roc curve for the model
        no_skill = len(y[:, i] == 1) / len(y[:, i])
        print(no_skill)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--',
                 label='No Skill class:'+str(i))
        plt.plot(recall[i], precision[i], color=colors[i], lw=2, marker='.',
                 label='Precision-recall for class {0} (area = {1:0.2f})'''.format(i, average_precision[i]))
        plt.scatter(recall[i][ix], precision[i][ix], marker='o',
                    color='black', label='Best')  # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # show the plot
    plt.savefig(directory+fpath+'Precision_Recall_curve.jpg')
    plt.close()
