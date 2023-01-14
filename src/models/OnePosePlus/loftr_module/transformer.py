import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        attention="linear",
        kernel_fn="elu + 1",
        redraw_interval=1,
        d_kernel=None,
        rezero=None,
        norm_method="layernorm",
    ):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = (
            LinearAttention(
                self.dim,
                kernel_fn=kernel_fn,
                redraw_interval=redraw_interval,
                d_kernel=d_kernel,
            )
            if attention == "linear"
            else FullAttention()
        )
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        if norm_method == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_method == "instancenorm":
            self.norm1 = nn.InstanceNorm1d(d_model)
            self.norm2 = nn.InstanceNorm1d(d_model)
        else:
            raise NotImplementedError

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if rezero is not None:
            self.res_weight = nn.Parameter(torch.Tensor([rezero]), requires_grad=True)
        self.rezero = True if rezero is not None else False

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.dropout1(
            message
        )  # dropout before merging multi-head queried outputs
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.dropout2(message)
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message if not self.rezero else x + self.res_weight * message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config["d_model"]  # Feature of query image
        self.nhead = config["nhead"]
        self.layer_names = list(config["layer_names"]) * config["layer_iter_n"]
        self.norm_method = config["norm_method"]
        if config["redraw_interval"] is not None:
            assert (
                config["redraw_interval"] % 2 == 0
            ), "redraw_interval must be divisible by 2 since each attetnion layer is repeatedly called twice."

        encoder_layer = build_encoder_layer(config)

        module_list = []
        for layer_name in self.layer_names:
            if layer_name in ["self", "cross"]:
                module_list.append(copy.deepcopy(encoder_layer))
            else:
                raise NotImplementedError
        self.layers = nn.ModuleList(module_list)

        if config["final_proj"]:
            self.final_proj = nn.Linear(config["d_model"], config["d_model"], bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, desc3d_db, desc2d_query, query_mask=None, return_middle_layer_features=False):
        """
        Args:
           desc3d_db (torch.Tensor): [N, C, L] 
           desc2d_db (torch.Tensor): [N, C, M]
           desc2d_query (torch.Tensor): [N, P, C]
           query_mask (torch.Tensor): [N, P]
           keypoints3D (torch.Tensor): [N, L, 3]
           desc2d_db_pad_mask (torch.Tensor): [N, M]
        """
        self.device = desc3d_db.device

        desc3d_db = torch.einsum("bdn->bnd", desc3d_db)  # [N, L, C]

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == "self":
                src0, src1 = desc2d_query, desc3d_db
                desc2d_query, desc3d_db = (
                    layer(desc2d_query, src0, query_mask, query_mask),
                    layer(desc3d_db, src1),
                )
            elif name == "cross":
                src0, src1 = desc3d_db, desc2d_query  # [N, L, C], [N, P, C]
                desc2d_query, desc3d_db = (
                    layer(desc2d_query, src0, x_mask=query_mask),
                    layer(desc3d_db, src1, source_mask=query_mask),
                )
            else:
                raise NotImplementedError
            
            if return_middle_layer_features:
                if i == len(self.layer_names) - 2:
                    desc2d_query_middle = desc2d_query.clone().detach()
                    desc3d_db_middle = desc3d_db.clone().detach()

        if return_middle_layer_features:
            return desc3d_db, desc2d_query, desc3d_db_middle, desc2d_query_middle
        else:
            return desc3d_db, desc2d_query


    def buildAdjMatrix(self, num_2d, num_3d):
        num_leaf = int(num_2d / num_3d)

        adj_matrix = torch.zeros(num_3d, num_2d).to(self.device)
        for i in range(num_3d):
            adj_matrix[i, num_leaf * i : num_leaf * (i + 1)] = 1 / num_leaf
        return adj_matrix


def build_encoder_layer(config):
    if config["type"] == "LoFTR":
        layer = LoFTREncoderLayer(
            config["d_model"],
            config["nhead"],
            config["dropout"],
            config["attention"],
            config["kernel_fn"],
            config["redraw_interval"],
            config["d_kernel"],
            rezero=config["rezero"],
            norm_method=config["norm_method"],
        )
    else:
        raise ValueError()
    return layer
