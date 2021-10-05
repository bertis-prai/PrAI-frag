import numpy as np
from sklearn.linear_model import LinearRegression


def get_ert_gap_from_irt_in_one_batch(pred_iRTs, true_eRTs, eRT1s, eRT2s):
    total_batch_gap = []
    pred_eRTs = []
    for pred_iRT, true_eRT, eRT1, eRT2 in zip(pred_iRTs, true_eRTs, eRT1s, eRT2s):
        pred_eRT = get_ert_from_irt(pred_iRT,eRT1,eRT2)
        pred_eRTs.append(pred_eRT)
        total_batch_gap.append(abs(true_eRT-pred_eRT))
    return np.asarray(total_batch_gap).mean(),np.asarray(pred_eRTs)

def get_ert_from_irt(pred_iRT, eRT1, eRT2, X=([[50], [70]])):
    pred_iRT = pred_iRT * 100
    y = np.asarray([eRT1,eRT2])
    reg = LinearRegression().fit(X,y)
    pred_eRT = reg.predict([[pred_iRT]])
    return pred_eRT