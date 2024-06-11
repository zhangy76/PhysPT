import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import config
import constants
from assets.utils import combine_output, rot6d_to_rotmat


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='gelu', norm_type=None, num_norm_groups=16):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.norm_type = norm_type
        self.affine_layers = nn.ModuleList()
        if norm_type is not None:
            self.norm_layers = nn.ModuleList()

        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            if norm_type == 'group_norm':
                self.norm_layers.append(nn.GroupNorm(num_norm_groups, nh))
            last_dim = nh

    def forward(self, x):
        for i, affine in enumerate(self.affine_layers):
            x = affine(x)
            if self.norm_type is not None:
                if len(x.shape) == 3 and self.norm_type == 'group_norm':
                    x = self.norm_layers[i](x.transpose(-1, -2)).transpose(-1, -2)
                else:
                    x = self.norm_layers[i](x)
            if i == (len(self.affine_layers) - 1):
                continue
            x = self.activation(x)
        return x


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PhysPT(nn.Module):

    def __init__(self, device,
                 seqlen,
                 mode: str,
                 f_dim: int,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 dropout: float = 0.1):
        super().__init__()

        self.device = device
        self.mode = mode

        self.f_dim = f_dim
        self.d_model = d_model

        self.start = torch.zeros([1, 1, d_model]).to(device)
        self.src_mask = generate_square_subsequent_mask(seqlen).to(device)

        self.encoder = MLP(self.f_dim, hidden_dims=(d_model, d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

        self.decoder = MLP(d_model, hidden_dims=(d_model, 144 + 3,))
        self.force_decoder = MLP(d_model, hidden_dims=(d_model, 29 * 3 + 69 + 6,))

    def forward(self, src: Tensor, tgt: Tensor, gt_shape, gt_gender_id, smplh_m, smplh_f, smpl) -> Tensor:

        src_embedding = self.encoder(src) * math.sqrt(self.d_model)

        batch_size = src_embedding.shape[1]
        src_embedding = self.pos_encoder(src_embedding)

        z_embedding = self.transformer_encoder(src_embedding)

        if self.mode == 'train':

            tgt_embedding = self.encoder(tgt)
            tgt_embedding = torch.cat([self.start.repeat(1, batch_size, 1), tgt_embedding[:-1, :]],
                                      dim=0) * math.sqrt(self.d_model)

            tgt_embedding = self.pos_encoder(tgt_embedding)

            output = self.transformer_decoder(tgt=tgt_embedding,
                                              memory=z_embedding,
                                              tgt_mask=self.src_mask)

            pred_q = self.decoder(output)
        else:
            tgt_embedding = self.start.clone().repeat(1, batch_size, 1)

            pred_dynamicinput = torch.zeros([config.seqlen, batch_size, self.f_dim]).float().to(self.device)
            pred_q = torch.zeros([config.seqlen, batch_size, 144 + 3]).float().to(self.device)
            for t in range(config.seqlen):

                tgt_embedding = self.pos_encoder(tgt_embedding * math.sqrt(self.d_model))
                tgt_mask = generate_square_subsequent_mask(tgt_embedding.shape[0]).to(self.device)

                output = self.transformer_decoder(tgt=tgt_embedding,
                                                  memory=z_embedding.clone(),
                                                  tgt_mask=tgt_mask)

                pred_q[t:t + 1] = self.decoder(output[t:t + 1])

                pred_pose_6d = pred_q[t:t + 1, :, 3:].clone()
                pred_rotmat_6d = rot6d_to_rotmat(pred_pose_6d).reshape([-1, 24, 3, 3])

                if gt_gender_id is None:
                    pred_output_smpl = smpl.forward(betas=gt_shape.reshape([-1, 10]), rotmat=pred_rotmat_6d)
                    pred_joints_smpl = pred_output_smpl.joints_smpl
                    pred_joints_regressor = pred_output_smpl.joints[:, constants.target]
                    pred_dynamicinput[t] = torch.cat(
                        [pred_q[t], pred_joints_smpl.reshape([-1, 72]), pred_joints_regressor.reshape([-1, 60])], dim=1)
                else:
                    pred_output_smplh_m = smplh_m.forward(betas=gt_shape[gt_gender_id == 0].reshape([-1, 10]),
                                                          rotmat=pred_rotmat_6d[gt_gender_id == 0])
                    pred_output_smplh_f = smplh_f.forward(betas=gt_shape[gt_gender_id == 1].reshape([-1, 10]),
                                                          rotmat=pred_rotmat_6d[gt_gender_id == 1])
                    pred_output_smpl = smpl.forward(betas=gt_shape[gt_gender_id == 2].reshape([-1, 10]),
                                                    rotmat=pred_rotmat_6d[gt_gender_id == 2])

                    pred_joints_smpl = combine_output(pred_output_smplh_m.joints_smpl, pred_output_smplh_f.joints_smpl,
                                                      pred_output_smpl.joints_smpl,
                                                      gt_gender_id, [batch_size, 24, 3], self.device)
                    pred_joints_regressor = combine_output(pred_output_smplh_m.joints[:, constants.target],
                                                           pred_output_smplh_f.joints[:, constants.target],
                                                           pred_output_smpl.joints[:, constants.target],
                                                           gt_gender_id, [batch_size, 20, 3], self.device)

                    pred_dynamicinput[t] = torch.cat(
                        [pred_q[t], pred_joints_smpl.reshape([-1, 72]), pred_joints_regressor.reshape([-1, 60])], dim=1)

                if t < (config.seqlen - 1):
                    tgt_embedding = torch.cat([self.start.clone().repeat(1, batch_size, 1),
                                               self.encoder(pred_dynamicinput[:t + 1].clone())], dim=0)

        pred_springmass = self.force_decoder(z_embedding).transpose(0, 1)
        pred_springmass[:, :, :29 * 3] = torch.tanh(pred_springmass[:, :, :29 * 3]) * 500 + 510

        return pred_q.transpose(0, 1), pred_springmass
