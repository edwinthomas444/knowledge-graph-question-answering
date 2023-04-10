from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

def multi_label_metrics(y_true, y_pred, labels, thresh):
    y_preds=[]
    for sample in y_pred:
        y_preds.append([1 if i>=thresh else 0 for i in sample])

    y_preds = np.array(y_preds).astype(int)
    y_true = np.array(y_true).astype(int)
    
    clf_report = classification_report(y_true, y_preds, target_names=labels, output_dict = True)

    # save and view csv report with class wise precision, recall and f1
    precision, recall, f1, support, label_names = [], [], [], [], []
    for cls in clf_report:
        metric_vals = clf_report[cls]
        precision.append(metric_vals['precision'])
        recall.append(metric_vals['recall'])
        f1.append(metric_vals['f1-score'])
        support.append(metric_vals['support'])
        label_names.append(cls)
    
    results = pd.DataFrame(data={
        'label':label_names,
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'support':support
    }, columns=['label','precision','recall','f1','support'])

    micro_avg_f1 = results['f1'].iloc[-4]
    
    return results, micro_avg_f1