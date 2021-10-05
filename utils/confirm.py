import os
import torch
from collections import OrderedDict
import csv
import shutil

def get_initial_checkpoint(config, new_check_dir, name=None):
    checkpoint_dir = os.path.join("./logs", config.RECIPE_DIR, new_check_dir)
    if name:
        return os.path.join(checkpoint_dir, name+'.pth')
    else:
        checkpoints = [checkpoint
                       for checkpoint in os.listdir(checkpoint_dir)
                       if checkpoint.startswith('top_') and checkpoint.endswith('.pth')]
        if checkpoints:
            print( list(sorted(checkpoints))[0])
            return os.path.join(checkpoint_dir, list(sorted(checkpoints))[0])
    return None

#
# def get_initial_sub_checkpoint(config):
#     checkpoint_dir = os.path.join(config.SUB_DIR, config.CHECKPOINT_SUB_DIR)
#     checkpoints = [checkpoint
#                    for checkpoint in os.listdir(checkpoint_dir)
#                    if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
#     if checkpoints:
#         return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
#     return None
#
# def load_checkpoint(config, model, checkpoint):
#     print('load checkpoint from', checkpoint)
#     model = torch.load(checkpoint)
#     return model

def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val
