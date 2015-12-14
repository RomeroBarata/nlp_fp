def precision(true_positive, false_positive):
    if (true_positive + false_positive) != 0:
        return true_positive / (true_positive + false_positive)
    else:
        return 1

def recall(true_positive, false_negative):
    if (true_positive + false_negative) != 0:
        return true_positive / (true_positive + false_negative)
    else:
        return 0

def f1(true_positive, false_positive, false_negative):
    prec = precision(true_positive, false_positive)
    rec = recall(true_positive, false_negative)
    if (prec + rec) != 0:
        return (2 * prec * rec) / (prec + rec)
    else:
        return 0

def accuracy(true_positive, true_negative, false_positive, false_negative):
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

def compute_metrics(true_class, predicted_class, positive_class):
    ''' Function to compute assessment metrics. These include: precision, recall, 
        f1, accuracy, true positive, true negative, false positive, false negative, and
        also it returns the name of the class of interest as well.
        Args:
                   true_class: A list containing the true class of each instance.
              predicted_class: A list of predictions made by a classifier.
               positive_class: The concept of interest of the classifier which made
                               the predictions.
    '''
    tp = tn = fp = fn = 0
    for i in range(len(predicted_class)):
        if predicted_class[i] == true_class[i]:
            if predicted_class[i] == positive_class:
                tp += 1
            else:
                tn += 1
        elif predicted_class[i] == positive_class:
            fp += 1
        else:
            fn += 1

    return {'Precision': precision(tp, fp), 'Recall': recall(tp, fn), 'F1': f1(tp, fp, fn), 'Accuracy': accuracy(tp, tn, fp, fn),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Positive Class': positive_class}
