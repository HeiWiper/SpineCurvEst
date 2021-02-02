import numpy as np
import pandas as pd

## find the SMAPE error
def smape(y_true, y_pred):
    numerator = np.sum(np.abs(y_true-y_pred), axis=1)
    denominator = np.sum(np.abs(y_true+y_pred), axis=1)
    smape_val = np.mean(numerator/ denominator) * 100
    return smape_val



true_csv_path = "../boostnet_labeldata/labels/test/angles.csv"
pred_csv_path = "angles_predicted.csv"
true_angles = pd.read_csv(true_csv_path, header=None).values
pred_angles = pd.read_csv(pred_csv_path, header=None).values
print (smape(true_angles, pred_angles))



