from math import sqrt, log

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
    
class FinePreprocess(nn.Module):
    def __init__(self, config, cf_res=None, feat_ids=None, feat_dims=None):
        super().__init__()
        self.config = config
        self.W = self.config['window_size']  # window size of fine-level
        
        self.coarse_id, self.fine_id = [int(log(r, 2)) for r in cf_res]  # coarse, fine resolutions
        self.feat_ids = feat_ids
        self.feat_dims = feat_dims  # dim of feats returned by backbone
        if self.feat_ids is None:
            assert self.feat_ids[0] > self.feat_ids[1]
            
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        
    def forward(self, data, feat_3D, feat_query_f):
        data.update({'W': self.W})
        if data['b_ids'].shape[0] == 0:
            feat_3D = torch.empty(0, self.config['d_model'], 1, device=feat_3D.device)
            feat_query_f = torch.empty(0, getattr(self, 'W_MAX', self.W)**2, self.config['d_model'], device=feat_3D.device)
            return feat_3D, feat_query_f
        
        return self._forward(data, feat_3D, feat_query_f)
    
    def _forward(self, data, feat_3D, feat_query_f):
        W = self.W
        stride = data['q_hw_f'][0] // data['q_hw_c'][0]
        feat_3D = feat_3D.permute(0,2,1) # B*N*C
            
        # unfold(crop) all local windows
        feat_query_f_unfold = F.unfold(feat_query_f, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_query_f_unfold = rearrange(feat_query_f_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        # select only the predicted matches
        feat_3D = feat_3D[data['b_ids'], data['i_ids'], :].unsqueeze(-1) # N*C*1

        feat_query_f_unfold = feat_query_f_unfold[data['b_ids'], data['j_ids']] # N*WW*C

        return feat_3D, feat_query_f_unfold