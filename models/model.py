import math
import torch
from torch import nn

from core.config import config
from models.resnetEncoder import ResNetEncoder
from models.Transformer import TransformerModel


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))  # 8x

    def forward(self, x, position_ids=None):
        position_embeddings = self.position_embeddings
        return x + position_embeddings


class Model(nn.Module):
    """
    model
    """
    def __init__(self, img_dim, patch_dim, embedding_dim, num_heads, num_layers,
                 dropout_rate=0.0, attn_dropout_rate=0.0, positional_encoding_type="learned"):
        super().__init__()
        self.resnet_encoder = ResNetEncoder()
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.num_patches = math.prod(math.ceil(dim / patch_dim) for dim in img_dim)
        self.seq_length = self.num_patches
        self.hidden_dim = self.num_patches
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            self.embedding_dim,
            num_layers,
            num_heads,
            self.hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim + 60, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 180)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim + 4*60, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64 + 28, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1))

    def forward(self, x, x_def, info):
        x_f = self.resnet_encoder(x)
        x_def = x_def[:, 0, :]

        x_feature = x_f

        x_f = x_f.permute(0, 2, 3, 4, 1).contiguous()
        x_f = x_f.view(x.size(0), -1, self.embedding_dim)

        x_f = self.position_encoding(x_f)
        x_f = self.pe_dropout(x_f)

        # apply transformer
        x, intmd_x = self.transformer(x_f)
        x = self.pre_head_ln(x)
        # reshape
        x = self._reshape_output(x)
        x = self.avg_pool(x).flatten(start_dim=1)

        x = torch.cat((x, x_def), dim=1)

        x_def_post = self.projector(x)
        x = torch.cat((x, x_def_post), dim=1)

        x1 = self.mlp[1](self.mlp[0](x))
        x2 = self.mlp[3](self.mlp[2](x1))
        x2 = torch.cat([x2, info], dim=1)
        x3 = self.mlp[5](self.mlp[4](x2))
        x4 = self.mlp[6](x3)

        return x_def_post, x1, x2, x3, x4

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(math.ceil(self.img_dim[0] / self.patch_dim)),
            int(math.ceil(self.img_dim[1] / self.patch_dim)),
            int(math.ceil(self.img_dim[2] / self.patch_dim)),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class Model_teacher(nn.Module):
    """
    model
    """
    def __init__(self, img_dim, patch_dim, embedding_dim, num_heads, num_layers,
                 dropout_rate=0.0, attn_dropout_rate=0.0, positional_encoding_type="learned"):
        super().__init__()
        self.resnet_encoder = ResNetEncoder()
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.num_patches = math.prod(math.ceil(dim / patch_dim) for dim in img_dim)
        self.seq_length = self.num_patches
        self.hidden_dim = self.num_patches
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            self.embedding_dim,
            num_layers,
            num_heads,
            self.hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim + 4*60, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64 + 28, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1))

    def forward(self, x, x_def, info):
        x_ti = torch.chunk(x, 4, dim=1)

        x_list = []
        x1 = self.resnet_encoder(x_ti[0])
        x_list.append(x1)
        x2 = self.resnet_encoder(x_ti[1])
        x_list.append(x2)
        x3 = self.resnet_encoder(x_ti[2])
        x_list.append(x3)
        x4 = self.resnet_encoder(x_ti[3])
        x_list.append(x4)

        x_combined = torch.stack(x_list, dim=1)  
        x_combined = 0.25 * x_combined
        x_f = torch.sum(x_combined, dim=1)  

        x_feature = x_f

        x_f = x_f.permute(0, 2, 3, 4, 1).contiguous()
        x_f = x_f.view(x.size(0), -1, self.embedding_dim)

        x_f = self.position_encoding(x_f)
        x_f = self.pe_dropout(x_f)

        # apply transformer
        x, intmd_x = self.transformer(x_f)
        x = self.pre_head_ln(x)
        # reshape
        x = self._reshape_output(x)
        x = self.avg_pool(x).flatten(start_dim=1)

        x_def = x_def.flatten(start_dim=1)
        x = torch.cat((x, x_def), dim=1)

        x1 = self.mlp[1](self.mlp[0](x))
        x2 = self.mlp[3](self.mlp[2](x1))
        x2 = torch.cat([x2, info], dim=1)
        x3 = self.mlp[5](self.mlp[4](x2))
        x4 = self.mlp[6](x3)

        return x1, x2, x3, x4

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(math.ceil(self.img_dim[0] / self.patch_dim)),
            int(math.ceil(self.img_dim[1] / self.patch_dim)),
            int(math.ceil(self.img_dim[2] / self.patch_dim)),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


def OSnet(_pe_type="learned", teacher=False):
    if teacher:
        model = Model_teacher(
            img_dim=config.MODEL.INPUT_SIZE,
            patch_dim=8,
            embedding_dim=512,
            num_heads=config.MODEL.NUM_HEADS,
            num_layers=config.MODEL.NUM_LAYERS,
            dropout_rate=config.MODEL.DROPOUT_RATE,
            attn_dropout_rate=config.MODEL.ATTN_DROPOUT_RATE,
            positional_encoding_type=_pe_type,
        )
    else:
        model = Model(
            img_dim=config.MODEL.INPUT_SIZE,
            patch_dim=8,
            embedding_dim=512,
            num_heads=config.MODEL.NUM_HEADS,
            num_layers=config.MODEL.NUM_LAYERS,
            dropout_rate=config.MODEL.DROPOUT_RATE,
            attn_dropout_rate=config.MODEL.ATTN_DROPOUT_RATE,
            positional_encoding_type=_pe_type,
        )
    return model
