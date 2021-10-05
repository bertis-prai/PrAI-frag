import random
import os
import numpy as np
import pandas as pd
import itertools

import torch
import torch.utils.data
from pyteomics import mass
from pyteomics import electrochem

from sklearn.model_selection import KFold
from numpy.random import seed

seed=20

#seed(seed)
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
        for seq_item in seq.infer_list:
            record = [0] * 15  # 20
            for index, item in enumerate(seq_item):
                record[index]=ALPHABET[item]
            records.append(record)
    else:
        record = [0] * 15  # 20
        for index, item in enumerate(seq):
            record[index]=ALPHABET[item]
        records.append(record)

    seq_df_columns = []
    for i in range(1, 16):  # 21):
        seq_df_columns.append("s" + str(i))

    seq_df = pd.DataFrame(records, columns=seq_df_columns)

    return seq_df

def calc_mass_charge_ce(aaseq_list):
    charges = []
    ces = []
    leng = []
    pros = []
    for aaseq in aaseq_list.infer_list:
        aaseq = str(aaseq)
        leng.append(len(aaseq))
        chrge = electrochem.charge(aaseq, 2)
        ms = mass.calculate_mass(sequence=aaseq)
        mz =  ms / chrge
        ce = (mz -206.6118761) / 12.24329778
        charges.append(round(chrge))
        ces.append(ce)
        pros.append(aaseq.count('P'))
    aaseq_list['f_charge'] = charges
    aaseq_list['f_ce'] = ces
    aaseq_list['f_length'] = leng
    aaseq_list['f_proline'] = pros
    return aaseq_list


def data_divide(input_df,x_cols,y_cols,x_feat, kfold = 10, fold_num = 0): #input file in df, column index(names) of the x_cols, y_cols, and x_feature
    cv = KFold(n_splits=kfold, random_state=20, shuffle=True)
    look = cv.split(input_df)

    # print(look)
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
            # pd.concat([val_seq, val_x_feat, val_y], axis=1).to_csv('fold_2_val.csv', index=False)
        #pepseq = input_df['pepcharge']
    print(index, fold_num, train_x.shape,val_x.shape, train_x_feat2.shape,val_x_feat2.shape,)
    return train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, train_seq, val_seq



def data_divide_feat2_2d(input_df,x_cols,y_cols,x_feat, kfold = 10, fold_num = 0): #input file in df, column index(names) of the x_cols, y_cols, and x_feature
    cv = KFold(n_splits=kfold, random_state=20, shuffle=True)
    look = cv.split(input_df)

    # print(look)
    for index, (t, v) in enumerate(cv.split(input_df)):
        if fold_num == index:
            print(index,fold_num)
            train_cv = input_df.iloc[t]
            val_cv = input_df.iloc[v]

            train_x = train_cv.loc[:, x_cols]
            train_x_feat = train_cv.loc[:, x_feat]
            train_x_feat2 = window_as_feat_2d(train_x)
            train_y = train_cv.loc[:, y_cols]

            val_x = val_cv.loc[:, x_cols]
            val_x_feat = val_cv.loc[:, x_feat]
            val_x_feat2 = window_as_feat_2d(val_x)
            val_y = val_cv.loc[:, y_cols]
            val_seq = val_cv.loc[:, 'Pepseq']
            pd.concat([val_seq, val_x_feat, val_y], axis=1).to_csv('fold_2_val.csv', index=False)
        #pepseq = input_df['pepcharge']
    print(index, fold_num, train_x.shape,val_x.shape, train_x_feat2.shape,val_x_feat2.shape,)
    return train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y



def window_as_feat(input_seq):
    n = np.array(input_seq)
    zeroes = np.zeros((len(input_seq), 1))
    nc = np.concatenate((zeroes, n, zeroes), 1)
    n_win = []
    for index, row in enumerate(nc):
        #if (index < 10):
       for i in range(12):
           j = i + 4
           n_win.append(row[i:j])
    ar = np.array(n_win, dtype=np.uint8)
    arr = ar.reshape(len(input_seq),48)
    return arr




def window_as_feat_2d(input_seq):
    n = np.array(input_seq)
    n_win = []
    for i in range(len(n)):
        row = n[i]
        row_win = []
        for j in range(12):
            row_win.append(row[j:j+4])
        n_win.append(np.array(row_win, dtype=np.uint8))
    ar = np.array(n_win, dtype=np.uint8)
    # arr = np.expand_dims(ar, axis=1)
    return ar




def input_params(config, input_df_file, kfold=10, fold_num=7):
    df = pd.read_csv(input_df_file)
    if config.DEBUG:
        df = df[:10]
    print("total_data_size: "+str(len(df)))

    ##Define y value containing columns
    df_col_list = df.columns.to_list()
    ycols = [i for i in df_col_list if i.startswith('y')]
    # print(ycols)

    ### X value contianing columns
    X_PEPTIDE_SEQUENCE_COLUMN = []
    for item in range(1, 16, 1):
        X_PEPTIDE_SEQUENCE_COLUMN.append("s" + str(item))

    ### Define x feature containing columns
    #x_feat_names = ycols+X_PEPTIDE_SEQUENCE_COLUMN
    #x_feat_names = df.drop(ycols+X_PEPTIDE_SEQUENCE_COLUMN, axis=1)
    x_feat_names = [i for i in df_col_list if i.startswith('f')]
    print(x_feat_names)

    ### get k fold index (default is 10 fold, 0fold is base)###
    train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, train_seq, val_seq = \
        data_divide(df, X_PEPTIDE_SEQUENCE_COLUMN, ycols, x_feat_names, kfold=kfold, fold_num=fold_num) #config.N_FOLD
    #print(train_x_feat)
    #print(train_y)
    train_x = np.asarray(train_x.values, dtype=np.uint8)
    train_y = np.asarray(train_y.values, dtype=np.float)
    train_x_feat = np.asarray(train_x_feat.values, dtype=np.float)
    val_x = np.asarray(val_x.values, dtype=np.uint8)
    val_y = np.asarray(val_y.values, dtype=np.float)
    val_x_feat = np.asarray(val_x_feat.values, dtype=np.float)

    train_x = torch.cuda.LongTensor(train_x)
    train_x_feat = torch.cuda.FloatTensor(train_x_feat)
    train_x_feat2 = torch.cuda.FloatTensor(train_x_feat2)
    #xtrainl = torch.Tensor(xtrainl)
    train_y = torch.cuda.FloatTensor(train_y)
    val_x = torch.cuda.LongTensor(val_x)
    val_x_feat = torch.cuda.FloatTensor(val_x_feat)
    val_x_feat2 = torch.cuda.FloatTensor(val_x_feat2)
    val_y = torch.cuda.FloatTensor(val_y)
 #   y_valid = torch.LongTensor(y_valid)
    #print(train_x.shape, train_y.shape)
    #print(train_x_feat.shape, val_x_feat.shape)
    #print(train_x_feat2.shape, val_x_feat2.shape)
    return train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols, train_seq, val_seq



def input_params_feat2_2d(config, input_df_file, kfold=10, fold_num=7):
    df = pd.read_csv(input_df_file)
    if config.DEBUG:
        df = df[:10]
    print("total_data_size: "+str(len(df)))

    ##Define y value containing columns
    df_col_list = df.columns.to_list()
    ycols = [i for i in df_col_list if i.startswith('y')]
    #print(ycols)

    ### X value contianing columns
    X_PEPTIDE_SEQUENCE_COLUMN = []
    for item in range(1, 16, 1):
        X_PEPTIDE_SEQUENCE_COLUMN.append("s" + str(item))

    ### Define x feature containing columns
    #x_feat_names = ycols+X_PEPTIDE_SEQUENCE_COLUMN
    #x_feat_names = df.drop(ycols+X_PEPTIDE_SEQUENCE_COLUMN, axis=1)
    x_feat_names = [i for i in df_col_list if i.startswith('f')]
    print(x_feat_names)

    ### get k fold index (default is 10 fold, 0fold is base)###
    train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y = data_divide_feat2_2d(df, X_PEPTIDE_SEQUENCE_COLUMN, ycols, x_feat_names, kfold=kfold, fold_num=fold_num) #config.N_FOLD
    #print(train_x_feat)
    #print(train_y)
    train_x = np.asarray(train_x.values, dtype=np.uint8)
    train_y = np.asarray(train_y.values, dtype=np.float)
    train_x_feat = np.asarray(train_x_feat.values, dtype=np.float)
    val_x = np.asarray(val_x.values, dtype=np.uint8)
    val_y = np.asarray(val_y.values, dtype=np.float)
    val_x_feat = np.asarray(val_x_feat.values, dtype=np.float)

    train_x = torch.cuda.LongTensor(train_x)
    train_x_feat = torch.cuda.FloatTensor(train_x_feat)
    train_x_feat2 = torch.cuda.FloatTensor(train_x_feat2)
    #xtrainl = torch.Tensor(xtrainl)
    train_y = torch.cuda.FloatTensor(train_y)
    val_x = torch.cuda.LongTensor(val_x)
    val_x_feat = torch.cuda.FloatTensor(val_x_feat)
    val_x_feat2 = torch.cuda.FloatTensor(val_x_feat2)
    val_y = torch.cuda.FloatTensor(val_y)
 #   y_valid = torch.LongTensor(y_valid)
    #print(train_x.shape, train_y.shape)
    #print(train_x_feat.shape, val_x_feat.shape)
    #print(train_x_feat2.shape, val_x_feat2.shape)
    return train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols




#input_df_file = './input/infer_5mer.csv'
def infer_input_params(input_file):
    df = pd.read_csv(input_file)

    print("Infering_data_size: "+str(len(df)))
    onehot = seqtoonehot(df)
    feat = calc_mass_charge_ce(df)
    df = pd.concat([feat, onehot], axis=1)
    df_col_list = df.columns.to_list()
    x_seq_names = [i for i in df_col_list if i.startswith('s')]
    x_feat_names = [i for i in df_col_list if i.startswith('f')]
    infer_x = df.loc[:, x_seq_names]
    infer_x_feat = df.loc[:, x_feat_names]
    infer_x_feat2 = window_as_feat(infer_x)
    infer_x = torch.cuda.LongTensor(np.asarray(infer_x))
    infer_x_feat = torch.cuda.LongTensor(np.asarray(infer_x_feat))
    # infer_x_feat = torch.cuda.FloatTensor(np.asarray(infer_x_feat))
    infer_x_feat2 = torch.cuda.FloatTensor(infer_x_feat2)
    # print(infer_x_feat)
    return infer_x, infer_x_feat, df.infer_list, infer_x_feat2

#infer_input_params(input_df_file)





def infer_input_params_for_wdata(input_file):
    df = pd.read_csv(input_file)

    print("Infering_data_size: "+str(len(df)))
    df_col_list = df.columns.to_list()
    ycols = [i for i in df_col_list if i.startswith('y')]
    x_seq_names = [i for i in df_col_list if i.startswith('s')]
    x_feat_names = [i for i in df_col_list if i.startswith('f')]
    infer_x = df.loc[:, x_seq_names]
    infer_x_feat = df.loc[:, x_feat_names]
    infer_y = df.loc[:, ycols]
    infer_x = np.asarray(infer_x.values, dtype=np.uint8)
    infer_y = np.asarray(infer_y.values, dtype=np.float)
    infer_x_feat2 = window_as_feat(infer_x)
    infer_x = torch.cuda.LongTensor(np.asarray(infer_x))
    infer_x_feat = torch.cuda.LongTensor(np.asarray(infer_x_feat))
    infer_x_feat2 = torch.cuda.FloatTensor(infer_x_feat2)
    infer_y = torch.cuda.FloatTensor(infer_y)
    return infer_x, infer_x_feat, df.Pepseq, infer_x_feat2, infer_y




def infer_input_params_y(input_file):
    df = pd.read_csv(input_file)

    print("Infering_data_size: "+str(len(df)))
    df_col_list = df.columns.to_list()
    ycols = [i for i in df_col_list if i.startswith('y')]
    infer_y = df.loc[:, ycols]
    infer_y = np.asarray(infer_y.values, dtype=np.float)
    infer_y = torch.cuda.FloatTensor(infer_y)
    return infer_y




def input_params_2(config, input_file, input_file_2, kfold=10):
    df = pd.read_csv(input_file)
    df2 = pd.read_csv(input_file_2)

    df_col_list = df.columns.to_list()
    ycols = [i for i in df_col_list if i.startswith('y')]
    x_seq_names = [i for i in df_col_list if i.startswith('s')]
    x_feat_names = [i for i in df_col_list if i.startswith('f')]

    onehot = seqtoonehot(df2)
    feat = calc_mass_charge_ce(df2)
    temp = pd.concat([feat, onehot], axis=1)

    df_x = df.loc[:, x_seq_names]
    df_x_feat = df.loc[:, x_feat_names]
    df_y = df.loc[:, ycols]

    df2_x = temp.loc[:, x_seq_names]
    df2_x_feat = temp.loc[:, x_feat_names]
    df2_y = df2.loc[:, ycols]

    df_x = pd.concat([df_x, df2_x], axis=0)
    df_x_feat = pd.concat([df_x_feat, df2_x_feat], axis=0)
    df_y = pd.concat([df_y, df2_y], axis=0)
    df = pd.concat([df_x, df_x_feat, df_y], axis=1)

    ### get k fold index (default is 10 fold, 0fold is base)###
    train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y = data_divide(df, x_seq_names, ycols, x_feat_names, kfold=kfold, fold_num=config.N_FOLD)
    #print(train_x_feat)
    #print(train_y)
    train_x = np.asarray(train_x.values, dtype=np.uint8)
    train_y = np.asarray(train_y.values, dtype=np.float)
    train_x_feat = np.asarray(train_x_feat.values, dtype=np.float)
    val_x = np.asarray(val_x.values, dtype=np.uint8)
    val_y = np.asarray(val_y.values, dtype=np.float)
    val_x_feat = np.asarray(val_x_feat.values, dtype=np.float)

    train_x = torch.cuda.LongTensor(train_x)
    train_x_feat = torch.cuda.FloatTensor(train_x_feat)
    train_x_feat2 = torch.cuda.FloatTensor(train_x_feat2)
    #xtrainl = torch.Tensor(xtrainl)
    train_y = torch.cuda.FloatTensor(train_y)
    val_x = torch.cuda.LongTensor(val_x)
    val_x_feat = torch.cuda.FloatTensor(val_x_feat)
    val_x_feat2 = torch.cuda.FloatTensor(val_x_feat2)
    val_y = torch.cuda.FloatTensor(val_y)
 #   y_valid = torch.LongTensor(y_valid)
    #print(train_x.shape, train_y.shape)
    #print(train_x_feat.shape, val_x_feat.shape)
    #print(train_x_feat2.shape, val_x_feat2.shape)
    return train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols




def input_params_3(config, input_df_file, kfold=10, fold_num=7):
    df = pd.read_csv(input_df_file)
    if config.DEBUG:
        df = df[:10]
    print("total_data_size: "+str(len(df)))

    ##Define y value containing columns
    df_col_list = df.columns.to_list()
    ycols = [i for i in df_col_list if i.startswith('y')]
    #print(ycols)

    ### X value contianing columns
    X_PEPTIDE_SEQUENCE_COLUMN = []
    for item in range(1, 16, 1):
        X_PEPTIDE_SEQUENCE_COLUMN.append("s" + str(item))

    ### Define x feature containing columns
    #x_feat_names = ycols+X_PEPTIDE_SEQUENCE_COLUMN
    #x_feat_names = df.drop(ycols+X_PEPTIDE_SEQUENCE_COLUMN, axis=1)
    x_feat_names = [i for i in df_col_list if i.startswith('f')]
    print(x_feat_names[:2])

    ### get k fold index (default is 10 fold, 0fold is base)###
    train_x, train_y, train_x_feat, val_x, val_x_feat, val_y = data_divide_3(df, X_PEPTIDE_SEQUENCE_COLUMN, ycols, x_feat_names[:2], kfold=kfold, fold_num=fold_num) #config.N_FOLD
    #print(train_x_feat)
    #print(train_y)
    train_x = np.asarray(train_x.values, dtype=np.uint8)
    train_y = np.asarray(train_y.values, dtype=np.float)
    train_x_feat = np.asarray(train_x_feat.values, dtype=np.float)
    val_x = np.asarray(val_x.values, dtype=np.uint8)
    val_y = np.asarray(val_y.values, dtype=np.float)
    val_x_feat = np.asarray(val_x_feat.values, dtype=np.float)

    train_x = torch.cuda.LongTensor(train_x)
    train_x_feat = torch.cuda.FloatTensor(train_x_feat)
    #xtrainl = torch.Tensor(xtrainl)
    train_y = torch.cuda.FloatTensor(train_y)
    val_x = torch.cuda.LongTensor(val_x)
    val_x_feat = torch.cuda.FloatTensor(val_x_feat)
    val_y = torch.cuda.FloatTensor(val_y)
 #   y_valid = torch.LongTensor(y_valid)
    #print(train_x.shape, train_y.shape)
    #print(train_x_feat.shape, val_x_feat.shape)
    #print(train_x_feat2.shape, val_x_feat2.shape)
    return train_x, train_y, train_x_feat, val_x, val_x_feat, val_y, ycols




def data_divide_3(input_df,x_cols,y_cols,x_feat, kfold = 10, fold_num = 0): #input file in df, column index(names) of the x_cols, y_cols, and x_feature
    cv = KFold(n_splits=kfold, random_state=20, shuffle=True)
    look = cv.split(input_df)

    # print(look)
    for index, (t, v) in enumerate(cv.split(input_df)):
        if fold_num == index:
            print(index,fold_num)
            train_cv = input_df.iloc[t]
            val_cv = input_df.iloc[v]

            train_x = train_cv.loc[:, x_cols]
            train_x_feat = train_cv.loc[:, x_feat]
            train_y = train_cv.loc[:, y_cols]

            val_x = val_cv.loc[:, x_cols]
            val_x_feat = val_cv.loc[:, x_feat]
            val_y = val_cv.loc[:, y_cols]
        #pepseq = input_df['pepcharge']
    print(index, fold_num, train_x.shape, val_x.shape)
    return train_x ,train_y, train_x_feat, val_x, val_x_feat, val_y

#infer_input_params(input_df_file)