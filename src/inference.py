import os
import pandas as pd
import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# import utils
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


def inference(config, new_check_dir):
    checkpoint = get_initial_checkpoint(config, new_check_dir)
    model = torch.jit.load('./logs/prai_frag/loss_mse_batchsize_128_foldNum_2/pre_train_model.zip')
    if torch.cuda.is_available():
        model.cuda()
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
        # print(prediction)

        prediction = prediction.cpu().numpy()

        pred_df = pd.DataFrame(prediction, columns=frag_list)
        pred_df = pd.concat([infer_df, pred_df], axis=1)
        # print(pred_df)
        print(config.INFER_DATA.replace('.csv', '_pred.csv'))
        pred_df.to_csv(config.INFER_DATA.replace('.csv', '_pred.csv'), index=False)


if __name__=="__main__":

    config = load('./src/config.yaml')
    fold = False

    if fold:
        bsize = config.EVAL.BATCH_SIZE
        lchoice = config.LOSS.NAME
        for i in range(10):
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
    else:
        bsize = config.EVAL.BATCH_SIZE
        lchoice = config.LOSS.NAME
        fNum = 2
        new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        inference(config, new_check_dir)