#Import
import os
import random
import numpy as np
import scipy.stats
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import median
from torch.utils.data import *
from datetime import datetime
from tensorboardX import SummaryWriter
import time
#### local Import
import utils
from utils.confirm import *
from utils.config import *
from utils.metrics import *
from utils.models import *
from utils.tools import *
import torch.optim.lr_scheduler as lr_scheduler

from dataset.dataset_windows import *

##basic settings

config = utils.config.load('./src/config.yaml')


torch.cuda.is_available() #cuda
device = torch.device("cuda") #or cpu
nowlog = datetime.now()
timelog = nowlog.strftime("%H%M")

seed_val = 20
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
os.environ['PYTHONHASHSEED'] = str(seed_val)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

lchoice = ""
bsize =""


def infer(config, batch, lchoice):
    checkpoint = utils.confirm.get_initial_checkpoint(config, new_check_dir )
    #model = PeptideIRTNet(config).cuda()
    #modelcheck = modelcheck.cuda()
    model = torch.load(checkpoint, map_location='cuda:0')
    model.eval()
    infer_x, infer_x_feat, aaseq, infer_x_feat2, infer_y = infer_input_params_for_wdata(config.INPUT_DATA)
    infer_data = TensorDataset(infer_x, infer_x_feat, infer_x_feat2, infer_y) #   val_x_feat,
    infer_data_sampler = SequentialSampler(infer_data)
    infer_dataloader = DataLoader( infer_data, sampler=infer_data_sampler, batch_size=batch)
    res_top2 = []
    res_top2id =[]
    data_top2 =[]
    data_top2id = []
    diffs = []
    df = pd.DataFrame()
    print(len(infer_x), len(aaseq))
    with torch.no_grad():
        for index, (infer_x, infer_x_feat, infer_x_feat2, infer_y) in enumerate(infer_dataloader):
            y_pred = model(infer_x, infer_x_feat, infer_x_feat2)
            #print(aaseq, y_pred)
            for i in range (len(infer_x)):
                top2list = infer_y.topk(2)[1][i].tolist()
                top2val = infer_y.topk(2)[0][i].tolist()
                ypred_max = y_pred.topk(1)[1][i].item()
                y_id = res_colnumto_yfrag( y_pred.topk(2)[1][i].tolist() )
                diff = y_pred.topk(2)[0][i].tolist()[0] - y_pred.topk(2)[0][i].tolist()[1]
            #    print(y_id, y_pred.topk(2)[0][i].tolist(),diff)
                data_top2id.append(res_colnumto_yfrag(top2list))
                data_top2.append(top2val)
                res_top2.append(y_pred.topk(2)[0][i].tolist())
                res_top2id.append(y_id)
                diffs.append(diff)
        print(len(infer_x), len(aaseq), len(diffs))
        df['seq'] = aaseq
        df['diff'] = diffs
        df['top2'] = res_top2id
        df['vals'] = res_top2
        df['top2_data_id'] = data_top2id
        df['vals_data'] = data_top2
    return df


inferopt = True
logb = []
logl = []
loghis =[]
if inferopt:

    bsize = 128
    lchoice = "mse"
    new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize)
    cond = "Running epoch " + str(bsize) + ", loss " + str(lchoice)
    # print("Running epoch " + str(bsize) + ", loss " + str(lchoice))
    #main(bsize, new_check_dir)
    # print("Confirm results for epoch " + str(bsize) + ", loss " + str(lchoice))
    res = infer(config, bsize, lchoice)


pd.set_option('display.width', None)
pd.set_option('display.max_columns', False)
pd.set_option('max_colwidth', None)
print(res.sort_values(by='diff', axis=0, ascending=False).head(30))
res.sort_values(by='diff', axis=0, ascending=False).to_csv("./logs/"+"infer_test_all_training_data"+".csv", index = False)