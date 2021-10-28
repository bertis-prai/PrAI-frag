from types import FrameType
import torch
import sys
sys.path.append('./')
import utils
from utils.confirm import *
from utils.config import *
from utils.models import *

from collections import OrderedDict

# path = "./logs/top_loss_0.0295_loss_0.016888_epoch_0032.pth"
# model = torch.load(path, map_location='cpu')

# print(model.state_dict())

# torch.save(model.state_dict(), "./logs/prai_frag/prai_frag_wight.pth")

config = utils.config.load('./src/config.yaml')
model = PeptideFragNet(config)
model.cuda()
# model.load_state_dict(torch.load("./logs/prai_frag/prai_frag_wight.pth"))

# weights = torch.load("./logs/prai_frag/prai_frag_wight.pth")

# # print(type(weights))
# # print(list(weights.keys()))

# print(model.state_dict().keys())
# new_keys = model.state_dict().keys()

# weights = OrderedDict(zip(list(model.state_dict().keys()), weights.values()))
# print(weights.keys())

# model.load_state_dict(weights)

# print(model.state_dict())
# torch.save(model.state_dict(), "./logs/prai_frag/loss_mse_batchsize_128_foldNum_2/prai_frag_wight_rename.pth")

# model.load_state_dict(torch.load('./logs/prai_frag/loss_mse_batchsize_128_foldNum_2/prai_frag_wight_rename.pth'))
# model.eval()

x = torch.ones(8,1,15)
feat1 = torch.ones(8,4)
feat2 = torch.ones(8,12,4)

print(model(x, feat1, feat2))

net_trace = torch.jit.trace(model, (x, feat1, feat2))
torch.jit.save(net_trace, './logs/prai_frag/loss_mse_batchsize_128_foldNum_2/model.zip')