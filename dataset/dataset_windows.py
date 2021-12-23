import random
import os
import math
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from pyteomics import mass
from pyteomics import electrochem

from sklearn.model_selection import KFold
from numpy.random import seed

seed=20

torch.cuda.manual_seed(seed)

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seqtoonehot(seq, is_multiple=True):
    ALPHABET = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "M(ox)": 21,
    }

    records = []
    if is_multiple:
        for seq_item in seq:
            seq_item = seq_item.replace(' ', '')
            record = [0] * 15  # 20
            for index, item in enumerate(seq_item):
                record[index]=ALPHABET[item]
            records.append(record)
    else:
        record = [0] * 15  # 20
        seq = seq.replace(' ', '')
        for index, item in enumerate(seq):
            record[index]=ALPHABET[item]
        records.append(record)

    seq_df_columns = []
    for i in range(1, 16):  # 21):
        seq_df_columns.append("s" + str(i))

    seq_df = pd.DataFrame(records, columns=seq_df_columns)

    return seq_df

def calc_mass_charge_ce(aaseq, charge, ce):
    leng = []
    pros = []

    for i in range(len(aaseq)):
        aaseq[i] = aaseq[i].replace(' ', '')
        seq = aaseq[i]
        if math.isnan(charge[i]): 
            charge[i] = electrochem.charge(seq, 2)
        if math.isnan(ce[i]): 
            ms = mass.calculate_mass(sequence=seq)
            mz = ms / charge[i]
            ce[i] = (mz -206.6118761) / 12.24329778
        charge[i] = round(charge[i])
        leng.append(len(seq))
        pros.append(seq.count('P'))
    
    df = pd.DataFrame(data=aaseq, columns=['Peptide'])
    df['f_charge'] = charge
    df['f_ce'] = ce
    df['f_length'] = leng
    df['f_proline'] = pros

    return df


def data_divide(input_df,x_cols,y_cols,x_feat, kfold = 10, fold_num = 0): #input file in df, column index(names) of the x_cols, y_cols, and x_feature
    cv = KFold(n_splits=kfold, random_state=20, shuffle=True)

    for index, (t, v) in enumerate(cv.split(input_df)):
        if fold_num == index:
            print(index,fold_num)
            train_cv = input_df.iloc[t]
            val_cv = input_df.iloc[v]

            train_x = train_cv.loc[:, x_cols]
            train_x_feat = train_cv.loc[:, x_feat]
            train_x_feat2 = window_as_feat(train_x)
            train_y = train_cv.loc[:, y_cols]
            train_seq = train_cv.loc[:, 'Pepseq']

            val_x = val_cv.loc[:, x_cols]
            val_x_feat = val_cv.loc[:, x_feat]
            val_x_feat2 = window_as_feat(val_x)
            val_y = val_cv.loc[:, y_cols]
            val_seq = val_cv.loc[:, 'Pepseq']
            
    return train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, train_seq, val_seq


def window_as_feat(input_seq):
    n = np.array(input_seq)
    n_win = []
    for i in range(len(n)):
        row = n[i]
        row_win = []
        for j in range(12):
            row_win.append(row[j:j+4])
        n_win.append(np.array(row_win, dtype=np.uint8))
    ar = np.array(n_win, dtype=np.uint8)
    return ar


def input_params(config, input_df_file, kfold=10, fold_num=2):
    df = pd.read_csv(input_df_file)
    if config.DEBUG:
        df = df[:10]
    print("total_data_size: "+str(len(df)))

    ##Define y value containing columns
    df_col_list = df.columns.to_list()
    ycols = [i for i in df_col_list if i.startswith('y')]

    ### X value contianing columns
    X_PEPTIDE_SEQUENCE_COLUMN = []
    for item in range(1, 16, 1):
        X_PEPTIDE_SEQUENCE_COLUMN.append("s" + str(item))

    ### Define x feature containing columns
    x_feat_names = [i for i in df_col_list if i.startswith('f')]

    ### get k fold index (default is 10 fold, 0fold is base)###
    train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, train_seq, val_seq = \
        data_divide(df, X_PEPTIDE_SEQUENCE_COLUMN, ycols, x_feat_names, kfold=kfold, fold_num=fold_num) #config.N_FOLD

    train_x = np.asarray(train_x.values, dtype=np.uint8)
    train_y = np.asarray(train_y.values, dtype=np.float)
    train_x_feat = np.asarray(train_x_feat.values, dtype=np.float)
    val_x = np.asarray(val_x.values, dtype=np.uint8)
    val_y = np.asarray(val_y.values, dtype=np.float)
    val_x_feat = np.asarray(val_x_feat.values, dtype=np.float)

    if torch.cuda.is_available():
        train_x = torch.cuda.LongTensor(train_x)
        train_x_feat = torch.cuda.FloatTensor(train_x_feat)
        train_x_feat2 = torch.cuda.FloatTensor(train_x_feat2)
        train_y = torch.cuda.FloatTensor(train_y)
        val_x = torch.cuda.LongTensor(val_x)
        val_x_feat = torch.cuda.FloatTensor(val_x_feat)
        val_x_feat2 = torch.cuda.FloatTensor(val_x_feat2)
        val_y = torch.cuda.FloatTensor(val_y)
    else:
        train_x = torch.LongTensor(train_x)
        train_x_feat = torch.FloatTensor(train_x_feat)
        train_x_feat2 = torch.FloatTensor(train_x_feat2)
        train_y = torch.FloatTensor(train_y)
        val_x = torch.LongTensor(val_x)
        val_x_feat = torch.FloatTensor(val_x_feat)
        val_x_feat2 = torch.FloatTensor(val_x_feat2)
        val_y = torch.FloatTensor(val_y)

    return train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols, train_seq, val_seq



def infer_input_params(input_df):
    seq, charge, ce = input_df['Peptide'].tolist(), input_df['Charge'].tolist(), input_df['CE'].tolist()
    cal_df = calc_mass_charge_ce(seq, charge, ce)
    x_feat_names = [i for i in cal_df if i.startswith('f')]

    x = seqtoonehot(seq, is_multiple=True)
    x_feat = cal_df.loc[:, x_feat_names]

    x = np.asarray(x.values, dtype=np.uint8)
    x_feat = np.asarray(x_feat.values, dtype=np.float)
    x_feat2 = window_as_feat(x)

    x = torch.LongTensor(x)
    x_feat = torch.FloatTensor(x_feat)
    x_feat2 = torch.FloatTensor(x_feat2)

    cal_input_df = cal_df[['Peptide', 'f_charge', 'f_ce']]
    cal_input_df.columns = ['Peptide', 'Charge', 'CE']

    return x, x_feat, x_feat2, cal_input_df
