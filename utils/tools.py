import random
import os
import numpy as np
import pandas as pd

import torch
import torch.utils.data

from sklearn.model_selection import KFold

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_kfold_x_columns_y_columns(input_df,x_columns,y_columns, kfold = 10, fold_num = 0, ert_columns=['RT','RT2','RT4']):
    cv = KFold(n_splits=kfold, random_state=2020, shuffle=True)
    for index, (t, v) in enumerate(cv.split(input_df)):
        if fold_num == 0:
            train_cv = input_df.iloc[t]
            val_cv = input_df.iloc[v]

            train_x = train_cv.loc[:, x_columns]
            train_y = train_cv.loc[:, y_columns]
            train_ert = train_cv.loc[:, ert_columns]

            val_x = val_cv.loc[:, x_columns]
            val_y = val_cv.loc[:, y_columns]
            val_ert = val_cv.loc[:, ert_columns]

    return train_x,train_y,train_ert,val_x,val_y,val_ert



def get_y_value():
    ######## y value ##########
    nums = []
    intensity_column = []
    for index in range(1, 16, 1):
        for charge_num in range(1, 4):
            for column_index_ion in ['y']:  # ,'b']:
                column_index = column_index_ion + str(index)
                num = str(index - 1)
                if charge_num >= 2:
                    column_index = column_index + "^" + str(charge_num)
                intensity_column.append(column_index)
                nums.append(num)


    return intensity_column, nums


def res_colnumto_yfrag(colnum, is_multilist=True):
    COLNUM = {
        0 : "y1",
        1: "y1^2",
        2: "y1^3",
        3: "y2",
        4: "y2^2",
        5: "y2^3",
        6: "y3",
        7: "y3^2",
        8: "y3^3",
        9: "y4",
        10: "y4^2",
        11: "y4^3",
        12: "y5",
        13: "y5^2",
        14: "y5^3",
        15: "y6",
        16: "y6^2",
        17: "y6^3",
        18: "y7",
        19: "y7^2",
        20: "y7^3",
        21: "y8",
        22: "y8^2",
        23: "y8^3",
        24: "y9",
        25: "y9^2",
        26: "y9^3",
        27: "y10",
        28: "y10^2",
        29: "y10^3",
        30: "y11",
        31: "y11^2",
        32: "y11^3",
        33: "y12",
        34: "y12^2",
        35: "y12^3",
        36: "y13",
        37: "y13^2",
        38: "y13^3",
        39: "y14",
        40: "y14^2",
        41: "y14^3",

    }

    yfragid =[]
    if is_multilist:
        ne = len(colnum)
        yfrag = [0] * ne
        for i, num in enumerate(colnum):
            yfrag[i]=COLNUM[num]
        yfragid.append(yfrag)
    else:
        yfragid = COLNUM[colnum]

    return yfragid

