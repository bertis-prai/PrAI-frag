from typing import Optional
import torch
import torch.nn as nn

import random
import os
import numpy as np
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

class Multiply(nn.Module):
  def __init__(self):
    super(Multiply, self).__init__()

  def forward(self, tensors):
    result = torch.ones(tensors[0].size())
    if str(tensors[0].device) == 'cuda:0':
        result = result.cuda()
    for t in tensors:
        result *= t
    return result


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x : torch.Tensor, mask : Optional[torch.Tensor]=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)



class PeptideFragNet(nn.Module):
    def __init__(self, config):
        super(PeptideFragNet, self).__init__()

        hidden_size = config.MODEL.PARAMS.HIDDEN_SIZE
        self.seq_len = config.MODEL.PARAMS.MAX_SEQUENCE_LEN
        self.embedding = nn.Embedding(config.MODEL.PARAMS.INPUT_WORD_SIZE, config.MODEL.PARAMS.EMBED_SIZE)
        self.gru1 = nn.GRU(config.MODEL.PARAMS.EMBED_SIZE, hidden_size, bidirectional=True, batch_first=True)
        self.gru1_dropout = nn.Dropout(p=0.4)
        self.gru_attention = Attention(hidden_size * 2, config.MODEL.PARAMS.MAX_SEQUENCE_LEN)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size *2)
        self.featlinear = nn.Linear(4, 32 )

        self.feat2conv = nn.Sequential(
            nn.Conv2d(1, 48, (1, 4), stride=1), nn.ReLU(),
            nn.Conv2d(48, 128, (2, 1), stride=1), nn.LeakyReLU(0.3),
            nn.Conv2d(128, 14, (11, 1), stride=1),
            nn.Dropout(0.4),
        )
        self.feat2linear = nn.Linear(14, 224)

        self.multiply = Multiply()

        self.linear_relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.dropout = nn.Dropout(0.1)

        self.gru2 = nn.GRU(hidden_size * 4, hidden_size, bidirectional=True, batch_first=True)
        self.gru2_dropout = nn.Dropout(p=0.4)
        self.gru_attention2 = Attention(hidden_size * 2, config.MODEL.PARAMS.MAX_SEQUENCE_LEN)
        self.linear3 = nn.Linear(hidden_size * 2, hidden_size )
        self.prediction = nn.Linear(hidden_size, 42)
        self.initial()


    def initial(self):
        self.embedding.weight.data.uniform_(-0.05, 0.05)

        for name, param in self.gru1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        for name, param in self.gru2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    
    def forward(self, x, x_feat, x_feat2):
        x = x.long()
        x = x.squeeze()
        x_feat = x_feat.float()
        x_feat = x_feat.unsqueeze(0) 
        x_feat2 = x_feat2.float()
        x_feat2 = x_feat2.unsqueeze(1)

        h_embedding = self.embedding(x)
        h_gru1, _h1 = self.gru1(h_embedding)
        h_gru1 = self.gru1_dropout(h_gru1)
        h_gru_atten = self.gru_attention(h_gru1)
        hgrudrop = self.dropout(h_gru_atten)
        hgrudrop = self.leaky_relu(hgrudrop)
        h_gru_lin = self.linear2(hgrudrop)

        x_feat = self.featlinear(x_feat)
        x_feat = torch.transpose(x_feat, 0, 1)
        x_feat = x_feat.squeeze(1) 

        x_feat2 = self.feat2conv(x_feat2)
        x_feat2 = x_feat2.squeeze()
        x_feat2 = x_feat2.unsqueeze(0) 

        x_feat2 = self.feat2linear(x_feat2) #
        x_feat2 = x_feat2.squeeze()
        x_feat=torch.cat([x_feat, x_feat2], dim=1)
        
        repeat_vector_input = self.multiply([h_gru_lin, x_feat])
        repeat_vector_input = repeat_vector_input.unsqueeze(1)
        repeat_vector_output = repeat_vector_input.repeat(1, self.seq_len, 2)
        hgru2, _h2 = self.gru2(repeat_vector_output, _h1)
        hgru2 = self.gru_attention2(hgru2)

        feature_extracted = self.dropout(hgru2)
        feature_extracted = self.leaky_relu(feature_extracted)
        feature_extracted = feature_extracted.squeeze(1)
        feature_extracted = self.linear3(feature_extracted)
        prediction = self.prediction(feature_extracted) 
        return prediction


class Multiply1(nn.Module):
  def __init__(self):
    super(Multiply1, self).__init__()

  def forward(self, tensors):
    result = torch.ones(tensors[0].size())
    if str(tensors[0].device) == 'cuda:0':
        result = result.cuda()
    for t in tensors:
        result *= t
    return result


class PeptideIRTNetConv2d(nn.Module):
    def __init__(self, config):
        super(PeptideIRTNetConv2d, self).__init__()

        hidden_size = config.MODEL.PARAMS.HIDDEN_SIZE
        self.seq_len = config.MODEL.PARAMS.MAX_SEQUENCE_LEN
        self.embedding = nn.Embedding(config.MODEL.PARAMS.INPUT_WORD_SIZE, config.MODEL.PARAMS.EMBED_SIZE)
        self.gru1 = nn.GRU(config.MODEL.PARAMS.EMBED_SIZE, hidden_size, bidirectional=True, batch_first=True)
        self.gru1_dropout = nn.Dropout(p=0.4)
        #self.gru2 = nn.GRU(hidden_size * 2, hidden_size , bidirectional=True, batch_first=True)
        #self.gru2_dropout = nn.Dropout(p=0.4)
        self.gru_attention = Attention(hidden_size * 2, config.MODEL.PARAMS.MAX_SEQUENCE_LEN)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size *2)
        self.featlinear = nn.Linear(4, 32 )

        self.feat2conv = nn.Sequential(
            nn.Conv2d(1, 48, (1, 4), stride=1), nn.ReLU(),
            nn.Conv2d(48, 128, (2, 1), stride=1), nn.LeakyReLU(0.3),
            nn.Conv2d(128, 14, (11, 1), stride=1),
            nn.Dropout(0.4),
        )
        self.feat2linear = nn.Linear(14, 224)

        self.multiply = Multiply1()

        #self.mult_lineared = nn.Linear(hidden_size * hidden_size, hidden_size)
        self.linear_relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.dropout = nn.Dropout(0.1)

        self.gru3 = nn.GRU(hidden_size * 4, hidden_size, bidirectional=True, batch_first=True)
        self.gru3_dropout = nn.Dropout(p=0.4)
        self.gru_attention3 = Attention(hidden_size * 2, config.MODEL.PARAMS.MAX_SEQUENCE_LEN)
        self.linear3 = nn.Linear(hidden_size * 2, hidden_size )
        self.prediction = nn.Linear(hidden_size, 42)
        self.initial()


    def initial(self):
        self.embedding.weight.data.uniform_(-0.05, 0.05)

        for name, param in self.gru1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        for name, param in self.gru3.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, x_feat, x_feat2):
        x = x.long()
        x = x.squeeze()
        x_feat = x_feat.float() #64,4
        x_feat = x_feat.unsqueeze(0) #64,4 --> 1,64,4
        x_feat2 = x_feat2.float() #64,48
        x_feat2 = x_feat2.unsqueeze(1) #64,48 --> 64,1,48

        h_embedding = self.embedding(x) ## batch X 15 X batch
        h_gru1, _h1 = self.gru1(h_embedding)  ## 32 x 15 X 512 ## 2,32,256
        h_gru1 = self.gru1_dropout(h_gru1)
        #h_gru2, _h2 = self.gru2(h_gru1, _h1)  ## 32 x 15 X 512 ##h2  # x (40x2 / bidirectional)
        #h_gru2 = self.gru2_dropout(h_gru2) ## 32 x 15 X 512
        h_gru_atten = self.gru_attention(h_gru1) # ## 32 x 15 X 512
        hgrudrop = self.dropout(h_gru_atten)
        hgrudrop = self.leaky_relu(hgrudrop)## 32 x 512
        h_gru_lin = self.linear2(hgrudrop) ## 32 x 256
        # gru_feature = self.linear_relu(h_gru_atten)
        #print("h_gur",h_gru_atten2.shape)
        #feature_extracted = self.leaky_relu(h_gru_atten) ## 80

        x_feat = self.featlinear(x_feat) #1,64,4 --> 1, 64,128
        #print("x_feat",x_feat.shape)
        x_feat = torch.transpose(x_feat, 0, 1) # 1,64,128 --> 64,1,128
       #print("x_feat_after",x_feat.shape)
        x_feat = x_feat.squeeze(1) #64,1,128 --> 64,128
        #x_feat2 = x_feat2.unsqueeze(1)#64,1,48

        x_feat2 = self.feat2conv(x_feat2) #64,1,48 --> 64,14,1, stride size matters for dim2
        x_feat2 = x_feat2.squeeze() #1,64,14
        x_feat2 = x_feat2.unsqueeze(0) #64,14,1 --> 1,64,14
        #x_feat2 = torch.transpose(x_feat2, 0, 1) #64,14,3 --> 14,64,3

        x_feat2 = self.feat2linear(x_feat2) #1,64,128
        x_feat2 = x_feat2.squeeze()
        x_feat=torch.cat([x_feat, x_feat2], dim=1)
        
        repeat_vector_input = self.multiply([h_gru_lin, x_feat])
        repeat_vector_input = repeat_vector_input.unsqueeze(1)
        repeat_vector_output = repeat_vector_input.repeat(1, self.seq_len, 2)
        hgru3, _h3 = self.gru3(repeat_vector_output, _h1)
        #hgru3 = self.gru3_dropout(hgru3)
        # print(hgru3.shape)
        hgru3 = self.gru_attention3(hgru3)
        # print(hgru3.shape)

        feature_extracted = self.dropout(hgru3)
        feature_extracted = self.leaky_relu(feature_extracted)## 80
        feature_extracted = feature_extracted.squeeze(1)
        feature_extracted = self.linear3(feature_extracted)
        prediction = self.prediction(feature_extracted) ## fcn out ## 80 x 16 x 1 -> last is 1
        return prediction