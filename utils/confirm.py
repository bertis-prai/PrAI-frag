import os
import torch

def get_initial_checkpoint(config, new_check_dir, name=None):
    checkpoint_dir = os.path.join("./logs", config.RECIPE_DIR, new_check_dir)
    if name:
        return os.path.join(checkpoint_dir, name+'.pth')
    else:
        checkpoints = [checkpoint
                       for checkpoint in os.listdir(checkpoint_dir) if checkpoint.endswith('.pth')]
                    #    if checkpoint.startswith('top_') and checkpoint.endswith('.pth')]
        if checkpoints:
            print(list(sorted(checkpoints))[0])
            return os.path.join(checkpoint_dir, list(sorted(checkpoints))[0])
    return None

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

# import sys
# sys.path.append('./')
# from utils.config import *
# config = load('./src/config.yaml')
# get_initial_checkpoint(config, 'loss_mse_batchsize_128_foldNum_2')