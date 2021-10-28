#Import
import os
import random
import numpy as np
import torch
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

    train_data = TensorDataset(train_x, train_x_feat, train_y, train_x_feat2 ) #   train_x_feat
    validation_data = TensorDataset(val_x, val_x_feat, val_y, val_x_feat2)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= batch)
    validation_sampler = SequentialSampler( validation_data)
    validation_dataloader = DataLoader( validation_data, sampler=validation_sampler, batch_size=batch)


    ###model
    model = torch.jit.load('./logs/prai_frag/loss_mse_batchsize_128_foldNum_2/model.zip')
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


if __name__=="__main__":
    fold = True
    bsize = config.EVAL.BATCH_SIZE
    lchoice = config.LOSS.NAME

    if fold:
        train_loss_hist = []
        val_loss_hist = []
        for i in range(10):
            fNum = i
            new_check_dir = "loss_" + str(lchoice) + "_batchsize_" + str(bsize) + "_foldNum_" + str(fNum)
            main(bsize, new_check_dir, fNum)
