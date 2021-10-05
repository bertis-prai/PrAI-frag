#Import
import os
import random
import numpy as np
from pandas.io import feather_format
import scipy.stats
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import median
# from torch.utils.data import *
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from datetime import datetime
from tensorboardX import SummaryWriter
import time
#### local Import
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils
from utils.confirm import *
from utils.config import *
from utils.metrics import *
from utils.models import *
import torch.optim.lr_scheduler as lr_scheduler
from torchviz import make_dot, make_dot_from_trace
from graphviz import render
from dataset.dataset_windows import *
# from dataset.dataset import *

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

#lchoice = "addl"
#lchoice =  "mse"
lchoice = ""
bsize =""


def eval_single_epoch(config, model, validation_dataloader, criterion, writer, epoch, fold_num):
    model.eval()
    with torch.no_grad():
        val_avg_loss = 0.
        val_avg_toploss = 0.
    for index, (val_x, val_x_feat, val_y, val_x_feat2) in enumerate(validation_dataloader):
        y_pred_val = model(val_x, val_x_feat, val_x_feat2)
        top3_val = val_y.topk(3)[1]
        columns_res3_val = y_pred_val.gather(-1, top3_val)
        val_toploss = criterion(columns_res3_val, val_y.topk(3)[0])

        loss_val = criterion(y_pred_val, val_y)
        val_avg_loss += loss_val / len(validation_dataloader)
        val_avg_toploss += val_toploss / len(validation_dataloader)


    writer.add_scalar('fold_num_{}/val/loss'.format(fold_num), val_avg_loss, epoch)
    writer.add_scalar('fold_num_{}/val/toploss'.format(fold_num), val_avg_toploss, epoch)

    return val_avg_loss, val_avg_toploss


def train_single_epoch(config, model, train_dataloader, criterion, optimizer, writer, epoch, fold_num):
    model.train()
    train_avg_loss = 0.
    total_ert_gap = 0.
    for index, (train_x, train_x_feat, train_y, train_x_feat2) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(train_x, train_x_feat, train_x_feat2)

        top3 = train_y.topk(3)[1]
        columns_res3 = y_pred.gather(-1, top3)
        train_avg_toploss = criterion(columns_res3, train_y.topk(3)[0])

        loss = criterion(y_pred, train_y)
        loss_add =0.3*loss + 0.7*train_avg_toploss
        if lchoice == "addl":
            loss_add.backward()
        else:
            loss.backward()
        train_avg_loss += loss.item()
        optimizer.step()

    train_avg_loss = train_avg_loss / len(train_dataloader)
    writer.add_scalar('fold_num_{}/train/lr'.format(fold_num), optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('fold_num_{}/train/loss'.format(fold_num), train_avg_loss, epoch)
    writer.add_scalar('fold_num_{}/train/toploss'.format(fold_num), train_avg_toploss, epoch)
    #.add_image('image', grid, 0)
    return train_avg_loss, train_avg_toploss


def train(config, model, train_dataloader, validation_dataloader, criterion, optimizer, scheduler, writer, checkpoint_dir, fold_num):

    min_score = np.inf
    start_time = time.time()
    n_epochs = config.TRAIN.NUM_EPOCHS
    for epoch in range(n_epochs):
        train_avg_loss, train_avg_toploss = train_single_epoch(config, model, train_dataloader, criterion, optimizer, writer, epoch, fold_num)
        val_avg_loss, val_avg_toploss = eval_single_epoch(config, model, validation_dataloader, criterion, writer, epoch, fold_num)
        val_addl6t4 = 0.6*val_avg_loss + 0.4*val_avg_toploss
        elapsed_time = time.time() - start_time
        print('[{}/{}] train_loss={:.4f} train_toploss_={:.4f} val_loss={:.4f} val_toploss_={:.4f} time={:.2f}s lr={:.8f}'.format(
            epoch + 1, n_epochs, train_avg_loss, train_avg_toploss, val_avg_loss, val_avg_toploss, elapsed_time, optimizer.param_groups[0]["lr"]))
        scheduler.step(val_avg_toploss)

        # train_loss_hist.append(train_avg_loss)
        # val_loss_hist.append(val_avg_loss)

        if min_score > val_avg_loss: #val_addl6t4
            min_score = val_avg_loss #val_addl6t4
            print(f"improved_score{val_avg_toploss}, save checkpoint")
            checkpoint_path = os.path.join(checkpoint_dir, 'top_loss_%.4f_loss_%04f_epoch_%04d.pth' % ( val_avg_toploss, val_avg_loss, epoch))
            torch.save(model, checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == config.TRAIN.EARLY_STOP_EPOCH:
                print(f'During {epochs_no_improve} epochs no improvement, So Early stopping!')
                break



def run(config: object, checkpoint_dir: object, batch: object, fold_num: object) -> object:
    writer = SummaryWriter(os.path.join('./logs', config.RECIPE_DIR))
    ##dataloading
    train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols, train_seq, val_seq = \
        input_params(config, config.INPUT_DATA, fold_num=fold_num)
    # train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols  = \
    #     input_params_feat2_2d(config, config.INPUT_DATA, fold_num=fold_num)
    # train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols = input_params_2(config, config.INPUT_DATA, config.INPUT_DATA_2)

    #train_data = TensorDataset(train_x, train_x_feat, train_y, train_x_feat2.reshape(-1,4,12) ) #   train_x_feat
    #validation_data = TensorDataset(val_x, val_x_feat, val_y, val_x_feat2.reshape(-1,4,12)) #   val_x_feat,

    train_data = TensorDataset(train_x, train_x_feat, train_y, train_x_feat2 ) #   train_x_feat
    validation_data = TensorDataset(val_x, val_x_feat, val_y, val_x_feat2)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= batch)
    validation_sampler = SequentialSampler( validation_data)
    validation_dataloader = DataLoader( validation_data, sampler=validation_sampler, batch_size=batch)


    ###model
    model = PeptideIRTNet3(config) ######
    # model = PeptideIRTNet4(config)
    # model = PeptideIRTNet5(config)
    # model = PeptideIRTNetConv2d(config)
    model = model.cuda()

    ###optimizer
    if config.OPTIMIZER.NAME == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.OPTIMIZER.LR)

    ###loss
    if config.LOSS.NAME == 'mse':
        criterion = torch.nn.MSELoss(reduction='mean')  # torch.nn.BCEWithLogitsLoss(reduction='mean')

    ### scheduler ###
    if config.SCHEDULER.NAME == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.SCHEDULER.PARAMS.MODE, factor=config.SCHEDULER.PARAMS.GAMMA, patience=config.SCHEDULER.PARAMS.PATIENCE)



    train(config, model, train_dataloader, validation_dataloader, criterion, optimizer, scheduler, writer, checkpoint_dir, fold_num)


def main(bsize, new_check_dir, fold_num):
    checkpoint_dir = os.path.join("./logs/", config.RECIPE_DIR, new_check_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    run(config, checkpoint_dir, bsize, fold_num)





def confirm(config, batch, lchoice, fNum):
    logr =[]
    without_minus = True #
    only_values = False
    checkpoint = utils.confirm.get_initial_checkpoint(config, new_check_dir)
    #model = PeptideIRTNet(config).cuda()
    #modelcheck = modelcheck.cuda()
    model = torch.load(checkpoint, map_location='cuda:0')
    model.eval()
    train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols, train_seq, val_seq = \
        input_params(config, config.INPUT_DATA, fold_num=fNum)
    # train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols = \
    #     input_params_feat2_2d(config, config.INPUT_DATA, fold_num=fNum)
    # train_x,train_y,train_x_feat,train_x_feat2, val_x,val_x_feat,val_x_feat2, val_y, ycols = input_params_2(config, config.INPUT_DATA, config.INPUT_DATA_2)
    # val_y = infer_input_params_y(config.INPUT_DATA)
    # val_x, val_x_feat, pep_seq, val_x_feat2 = infer_input_params(config.INPUT_DATA)
    
    val_x = torch.cat((train_x, val_x), 0)
    val_x_feat = torch.cat((train_x_feat, val_x_feat), 0)
    val_y = torch.cat((train_y, val_y), 0)
    val_x_feat2 = torch.cat((train_x_feat2, val_x_feat2), 0)

    # charge_list = val_x_feat.clone().cpu().numpy().astype(np.int)[:,0]

    # validation_data = TensorDataset(val_x, val_x_feat, val_y)
    validation_data = TensorDataset(val_x, val_x_feat, val_y, val_x_feat2)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch)
    tot = 0
    pcc_list = [] #
    y_pred_list = []
    val_y_list = []
    charge_list = []
    with torch.no_grad():
        top1tot = 0
        top1isintot = 0
        top3intop3tot = 0
        top2tot = 0
        top2intop3tot = 0
        # for index, (val_x, val_x_feat, val_y) in enumerate(validation_dataloader):
        #     y_pred = model(val_x, val_x_feat)
        for index, (val_x, val_x_feat, val_y, val_x_feat2) in enumerate(validation_dataloader):
            y_pred = model(val_x, val_x_feat, val_x_feat2)
            top1 = 0
            top2 = 0
            top1isin3 = 0
            top3 = val_y.topk(3)[1]
            # print(y_pred.shape, val_y.shape)
            # print(y_pred[1].max(), val_y[1].max())

            # if without_minus:
            #     pcc_y = val_y[1].tolist().copy()
            #     pcc_pred = y_pred[1].tolist().copy()
            #     if len(find_index(pcc_y, -1))==0:
            #         pcc = pearsonr(y_pred[1], val_y[1])
            #     else:
            #         for i in reversed(find_index(pcc_y, -1)):
            #             del pcc_y[i]
            #             del pcc_pred[i]
            #         pcc_y = torch.tensor(pcc_y)
            #         pcc_pred = torch.tensor(pcc_pred)
            #         pcc = pearsonr(pcc_pred, pcc_y)
            # else:
            #     pcc = pearsonr(y_pred[1], val_y[1])
            pcc = pearsonr(y_pred[1], val_y[1])
            # print(y_pred[1].shape, val_y[1].shape)
            columns_res3 = y_pred.gather(-1, top3)
            #print(index, train_y.topk(3)[0], columns_res3)
            for i in range (len(val_x)):
                val_y_list.append(val_y[i].cpu().numpy().astype(np.float))
                y_pred_list.append(y_pred[i].cpu().numpy().astype(np.float))
                charge_list.append([val_x_feat[i, 0].item()])
                
                tott = 1
                tot += tott
                ypred_max = y_pred.topk(1)[1][i].item()
                ypred_top2 = y_pred.topk(2)[1][i].tolist()
                ypred_top3 = y_pred.topk(3)[1][i].tolist()
                #print(y_pred[i])

                # pcc_each = pearsonr(y_pred[i], val_y[i]).tolist()

                top1list = val_y.topk(1)[1][i].tolist()
                top2list = val_y.topk(2)[1][i].tolist()
                top3list = val_y.topk(3)[1][i].tolist()


                if without_minus:
                    pcc_y = val_y[i].tolist().copy()
                    pcc_pred = y_pred[i].tolist().copy()
                    if len(find_index(pcc_y, -1))==0:
                        pcc_list.append(pearsonr(y_pred[i], val_y[i]).item())
                    else:
                        for i in reversed(find_index(pcc_y, -1)):
                            del pcc_y[i]
                            del pcc_pred[i]
                        pcc_y = torch.tensor(pcc_y)
                        pcc_pred = torch.tensor(pcc_pred)
                        y_pred_max = pcc_pred.max().item()
                        for j in range(len( pcc_pred)):
                            if pcc_pred[j] > 0:
                                pcc_pred[j] = pcc_pred[j] / y_pred_max
                        # pcc_each = pearsonr(pcc_pred, pcc_y).tolist()
                        pcc_list.append(pearsonr(pcc_pred, pcc_y).item())
                elif only_values:
                    pcc_y_over_zero = (val_y[i] > 0).type(torch.uint8).tolist()
                    pcc_y = val_y[i].tolist().copy()
                    pcc_pred = y_pred[i].tolist().copy()
                    if len(find_index(pcc_y_over_zero, 0))==0:
                        pcc_list.append(pearsonr(y_pred[i], val_y[i]).item())
                    else:
                        for i in reversed(find_index(pcc_y_over_zero, 0)):
                            del pcc_y[i]
                            del pcc_pred[i]
                        pcc_y = torch.tensor(pcc_y)
                        pcc_pred = torch.tensor(pcc_pred)
                        # print(len(pcc_y),len(pcc_pred))
                        pcc_list.append(pearsonr(pcc_pred, pcc_y).item())
                else:
                    # pcc_each = pearsonr(y_pred[i], val_y[i]).tolist()
                    y_pred_max = y_pred[i].max().item()
                    for j in range(len(y_pred[i])):
                        if y_pred[i][j] > 0:
                            y_pred[i][j] = y_pred[i][j] / y_pred_max
                    # print(y_pred[i])
                    pcc_list.append(pearsonr(y_pred[i], val_y[i]).item())
                # print(pcc_each)
                # print(set(ypred_top2).intersection(top3list), len(set(ypred_top2).intersection(top3list) ))

                if ypred_top2 == top2list:
                    top2 = 1
                    top2tot += top2
                else:
                    top3 = 0

                if len(set(ypred_top2).intersection(top3list)) == 2:
                     top2intop3 = 1
                     top2intop3tot += top2intop3
                elif len(set(ypred_top2).intersection(top3list)) == 1:
                     top2_onematch = 1
                else:
                     top2nomatch = 1
                if ypred_max in top1list:
                     top1 = 1
                     top1tot += top1
                else:
                     top1 = 0
                if ypred_max in top3list:
                    top1isin3 = 1
                    top1isintot += top1isin3
                else:
                    top1isin3 = 0

                if ypred_top3 == top3list:
                    top3 = 1
                    top3intop3tot += top3
                else:
                    top3 = 0

                #print(index, i, val_y.topk(3)[0][i], columns_res3[i], val_y.topk(3)[1][i].tolist(), y_pred.topk(3)[1][i].tolist(), top1, top1isin3)
    print(top1tot, top1isintot, tot)
    # pcc_med = np.median(pcc_each)
    # pcc_mean = np.mean(pcc_each)
    pcc_med = np.nanmedian(pcc_list)
    pcc_mean = np.nanmean(pcc_list)
    # pcc_df = pd.DataFrame({'pcc' : pcc_list})
    # pcc_df = pd.concat([pep_seq,pcc_df], axis=1)
    # pcc_df.colums = ['Pepseq', 'pcc']
    # feat_col = ['f_charge', 'f_ce', 'f_length', 'f_proline']
    # feat_df = pd.DataFrame(x_feat, columns=feat_col)
    # pcc_df = pd.concat([pcc_df, feat_df], axis=1)
    # pcc_df.to_csv("exp_pcc_feat.csv", index=False)
    # print(pcc_df)
    # print(np.where(np.array(pcc_list) < 0.85))
    # print(ycols)
    os.makedirs("./result/pcc_{}/".format(config.RECIPE_DIR), exist_ok=True)
    pep_seq = pd.concat([train_seq, val_seq], axis=0).reset_index(drop=True)
    pep_pcc = pd.DataFrame(pcc_list, columns=['pcc'])
    pep_pcc = pd.concat([pep_seq, pep_pcc], axis=1)
    pep_pcc.to_csv('./result/pcc_{}/{}_wom_rat_test.csv'.format(config.RECIPE_DIR, new_check_dir), index=False)
    
    charge_df = pd.DataFrame(charge_list, columns=['f_charge'])

    pep_true = pd.DataFrame(val_y_list, columns=ycols)
    pep_true = pd.concat([pep_seq, pep_true, charge_df], axis=1)

    pep_pred = pd.DataFrame(y_pred_list, columns=ycols)
    pep_pred = pd.concat([pep_seq, pep_pred, charge_df], axis=1)
    
    pep_true.to_csv('./result/pcc_{}/{}_wom_rat_test_t.csv'.format(config.RECIPE_DIR, new_check_dir), index=False)
    pep_pred.to_csv('./result/pcc_{}/{}_wom_rat_test_p.csv'.format(config.RECIPE_DIR, new_check_dir), index=False)
    # print(config.RECIPE_DIR[-])
    # np.savetxt(config.RECIPE_DIR[-:] +'_fnum_'+str(fNum)+'_pcc.csv', pcc_list, delimiter=',') 
    top1perc= 100 * top1tot / tot
    top1in3perc = 100 * top1isintot / tot
    top3perc = 100 * top3intop3tot / tot
    top2perc = 100 * top2tot / tot
    top2in3perc = 100 * top2intop3tot / tot
    percs = str(batch) + ", " + str(lchoice) + ", " + "{:.3f}".format(top1perc) + ", " + "{:.3f}".format(top1in3perc) + ", " + "{:.3f}".format(top2perc) + ", " + "{:.3f}".format(top2in3perc) + ", " + "{:.3f}".format(top3perc) + ", " + "{:.3f}".format(pcc) + ", " + "{:.3f}".format(pcc_med) + ", " + "{:.3f}".format(pcc_mean)

    logr.append(percs)
    #logag = "top1_match_accuracy={:.4f} top1_in_top3_accuracy={:.4f} top2_in_top2_accuracy={:.4f} top2_in_top3_accuracy={:.4f} top3_in_top3_accuracy={:.4f}".format(top1perc, top1in3perc, top2perc, top2in3perc , top3perc
    print(str(batch) + " " + str(lchoice) +" " + 'top1_match_accuracy={:.4f} top1_in_top3_accuracy={:.4f} top2_in_top2_accuracy={:.4f} top2_in_top3_accuracy={:.4f} top3_in_top3_accuracy={:.4f} pearson_corr={:.4f} pearson_corr_med={:.4f} pearson_corr_mean={:.4f}'.format(\
        top1perc, top1in3perc, top2perc, top2in3perc , top3perc, pcc, pcc_med, pcc_mean))
    print(logr)
    print(config.INPUT_DATA, pcc_med, pcc_mean)
    total_log.append([top1perc, top1in3perc, top2perc, top2in3perc , top3perc, pcc_med, pcc_mean])


###
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


debug = True
logb = []
logl = []
loghis =[]
fold = True
if debug:

    train_loss_hist = []
    val_loss_hist = []
    total_log = []

    if fold:
        for i in range(10):
            #print(i)
            if i == 0:
                bsize = 128
                lchoice = "mse"
                fNum = i
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            elif i == 1:
                bsize = 128
                lchoice = "mse"
                fNum = i
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            if i == 2:
                bsize = 128
                lchoice = "mse"
                fNum = i
                new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
                # confirm(config, bsize, lchoice, fNum)
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            elif i == 3:
                bsize = 128
                lchoice = "mse"
                fNum = i
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            elif i == 4:
                bsize = 128
                lchoice = "mse"
                fNum = i
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            elif i == 5:
                bsize = 128
                lchoice = "mse"
                fNum = i
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            elif i == 6:
                bsize = 128
                lchoice = "mse"
                fNum = i
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            elif i == 7:
                bsize = 128
                lchoice = "mse"
                fNum = i
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            elif i == 8:
                bsize = 128
                lchoice = "mse"
                fNum = i
                # new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
            elif i == 9:
                bsize = 128
                lchoice = "mse"
                fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
            confirm(config, bsize, lchoice, fNum)
        total_log = np.array(total_log)
        np.savetxt('./result/result_{}_wom_rat_test.csv'.format(config.RECIPE_DIR), total_log, delimiter=',')
    else:
        bsize = 128
        lchoice = "mse"
        new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) #+ "_foldNum_" + str(fNum)
        cond = "Running epoch " + str(bsize) + ", loss " + str(lchoice)
        # print("Running epoch " + str(bsize) + ", loss " + str(lchoice))
        # main(bsize, new_check_dir)
        # print("Confirm results for epoch " + str(bsize) + ", loss " + str(lchoice))

        confirm(config, bsize, lchoice, config.N_FOLD)

    # %%
    import matplotlib.pyplot as plt

    plt.plot(train_loss_hist)
    plt.show()

    # %%
    plt.plot(val_loss_hist)
    plt.show()

else:
    train_loss_hist = []
    val_loss_hist = []
    for i in range(10):
        print(i)
        if i == 0:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        elif i == 1:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        elif i == 2:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
            # main(bsize, new_check_dir, fNum)
        elif i == 3:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        elif i == 4:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        elif i == 5:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        elif i == 6:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        elif i == 7:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        elif i == 8:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
        elif i == 9:
            bsize = 128
            lchoice = "mse"
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)

        # elif i == 2:
        #     bsize=256
        #     lchoice = "mse"
        #     new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize)
        # elif i == 3:
        #     bsize=512
        #     lchoice = "mse"
        #     new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize)
        # elif i == 4:
        #     bsize = 32
        #     lchoice = "addl"
        #     new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize)
        # elif i == 5:
        #     bsize = 64
        #     lchoice = "addl"
        #     new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize)
        # elif i == 6:
        #     bsize = 256
        #     lchoice = "addl"
        #     new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize)
        # elif i == 7:
        #     bsize = 512
        #     lchoice = "addl"
        #     new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize)
        main(bsize, new_check_dir, fNum)

        # confirm(config, bsize, lchoice)
#print(loghis)
# testlog = pd.DataFrame()
# testlogb = pd.DataFrame(data=logb)
# testlogl = pd.DataFrame(data=logl)
# testlogr = pd.DataFrame(data=logr)
# testlog['batch'] = testlogb.values
# testlog['loss'] = testlogl.values
# testlog['res'] = testlogr.values
# #testlogb.merge(testlogl, testlogr.items)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', False)
pd.set_option('max_colwidth', None)

# print(testlogr)
# testlogr.to_csv("./logs/"+str(config.RECIPE_DIR)+".csv", index = False)
#logss = pd.DataFrame(loghis)
#print(logss)
# if checkres:
#     #main()
#     confirm(config, batch)
# else:
#     main()

