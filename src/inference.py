import os
import pandas as pd
import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.confirm import *
from utils.config import *

from dataset.dataset_windows import *


def find_index(data, target):
    res = []
    lis = data
    while True:
        try:
            res.append(lis.index(target) + (res[-1]+1 if len(res)!=0 else 0)) #+1의 이유 : 0부터 시작이니까
            lis = data[res[-1]+1:]
        except:
            break     
    return res


def parse_result(df):
    frag_list = []
    for i in range(1,15):
        frag_list.append('y{}'.format(i))
        for j in [2, 3]:
            frag_list.append('y{}^{}'.format(i, j))

    for i in range(len(df)):
        if df.at[i, 'Charge'] == 1:
            df.loc[i, [f for f in frag_list if ('^2' in f) or ('^3' in f)]] = np.nan
        elif df.at[i, 'Charge'] == 2:
            df.loc[i, [f for f in frag_list if '^3' in f]] = np.nan

        df.loc[i, [f for f in frag_list if int(f.split('^')[0].replace('y','')) >= len(df.at[i, 'Peptide'])]] = np.nan
    
    return df


def inference(config, new_check_dir, name):
    checkpoint = get_initial_checkpoint(config, new_check_dir, name=name)
    model = torch.jit.load(checkpoint)
    # if torch.cuda.is_available():
    #     model.cuda()
    model.eval()

    infer_df = pd.read_csv(config.INFER_DATA)

    x, x_feat, x_feat2, infer_df = infer_input_params(infer_df)
    
    frag_list = []
    for i in range(1,15):
        frag_list.append('y{}'.format(i))
        for j in [2, 3]:
            frag_list.append('y{}^{}'.format(i, j))
    
    with torch.no_grad():
        prediction = model(x, x_feat, x_feat2)

        prediction = prediction.cpu().numpy()
        prediction = np.where(prediction < 1e-4, 0, prediction)

        pred_df = pd.DataFrame(prediction, columns=frag_list)
        pred_df = pd.concat([infer_df, pred_df], axis=1)
        pred_df = parse_result(pred_df)

        pred_df.to_csv(config.INFER_DATA.replace('.csv', '_pred.csv'), index=False, float_format='%.4f')


if __name__=="__main__":

    config = load('./src/config.yaml')
    lchoice = config.LOSS.NAME

    if lchoice == 'mse':
        new_check_dir = "loss_" + str(lchoice)
        name='pre_train_model.zip'
    elif lchoice == 'addl':
        new_check_dir = "loss_" + str(lchoice)
        name='pre_train_add_loss_model.zip'

    inference(config, new_check_dir, name)
