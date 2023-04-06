import torch
from torch import nn
from torch.nn import functional as F

import deepmorph.lrp.lrp_layers as lrp_layers


class VectorQuantizer(nn.Module):
    '''Vector quantizer adapted from rosinality/vq-vae-2-pytorch
    '''
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            lrp_layers.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            lrp_layers.Conv2d(channel, in_channel, 1),
        )
        self.add = lrp_layers.Add()

    def forward(self, input):
        out = self.conv(input)
        return self.add.add(input, out)
    
    def lrp(self, R):
        R1, R2 = self.add.lrp(R)
        
        for c in self.conv[::-1]:
            if isinstance(c, lrp_layers.Conv2d):
                R2 = c.lrp(R2)
                
        return R1 + R2
    
class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                lrp_layers.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                lrp_layers.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                lrp_layers.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                lrp_layers.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                lrp_layers.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    
    def lrp(self, R):
        for c in self.blocks[::-1]:
            if isinstance(c, (lrp_layers.Conv2d, ResBlock)):
                R = c.lrp(R)
        
        return R
    
class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [lrp_layers.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    lrp_layers.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    lrp_layers.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                lrp_layers.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    
    def lrp(self, R):
        for c in self.blocks[::-1]:
            if isinstance(c, (lrp_layers.Conv2d, ResBlock, lrp_layers.ConvTranspose2d)):
                R = c.lrp(R)
        return R
    
class FCClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024):
        super().__init__()
        self.flatten = lrp_layers.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            lrp_layers.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            lrp_layers.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, X):
        X = self.flatten(X)
        return self.fc(X)
    
    def lrp(self, R):
        for l in self.fc[::-1]:
            if isinstance(l, lrp_layers.Linear):
                R = l.lrp(R)
        
        # The R is reshaped to the input shape       
        return R.reshape(self.flatten.input_shape)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        
        n_categories=1,
        img_xy_shape=(128, 128)
    ):
        super().__init__()

        # Encoding
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = lrp_layers.Conv2d(channel, embed_dim, 1)
        self.quantize_t = VectorQuantizer(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = lrp_layers.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = VectorQuantizer(embed_dim, n_embed)
        
        # Classification
        self.n_categories = n_categories
        if n_categories > 1:
            x_shape, y_shape = img_xy_shape
            self.classifier_t = FCClassifier(embed_dim * x_shape * y_shape // 64, 
                                             n_categories, hidden_dim=1024)
            self.classifier_b = FCClassifier(embed_dim * x_shape * y_shape // 16, 
                                             n_categories, hidden_dim=1024)
        
        # Decoding
        self.upsample_t = lrp_layers.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        class_t, class_b = self.classify(quant_t, quant_b)
        dec = self.decode(quant_t, quant_b)

        return dec, diff, class_t, class_b

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def classify(self, quant_t, quant_b):
        if self.n_categories == 1:
            return 0, 0
        return self.classifier_t(quant_t), self.classifier_b(quant_b)
    
    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
    
    def lrp(self, R):
        # LRP on the top route
        R_t = self.classifier_t.lrp(R)
        embed_dim = R_t.shape[1]
        R_t = self.quantize_conv_t.lrp(R_t)
        R_t = self.enc_t.lrp(R_t)
        R_t = self.enc_b.lrp(R_t)
                
        # LRP on the bottom route
        R_b = self.classifier_b.lrp(R)
        R_b = self.quantize_conv_b.lrp(R_b)
        R_bt = R_b[:, :embed_dim]
        R_bb = R_b[:, embed_dim:]
        R_bt = self.dec_t.lrp(R_bt)
        R_bt = self.quantize_conv_t.lrp(R_bt)
        R_bt = self.enc_t.lrp(R_bt)
        R_b = R_bb + R_bt
        R_b = self.enc_b.lrp(R_b)
        
        # Combine the relevance 
        return R_t + R_b
        
    