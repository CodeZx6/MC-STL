from functools import partial
import torch
import torch.nn as nn

from transformer.vit import PatchEmbed, Block

from helper.utils.pos_embed import get_2d_sincos_pos_embed

from helper.config import setup_seed
setup_seed(2021)
class MaskViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=32, patch_size=16, in_chans=20,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.in_chans = in_chans
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * self.in_chans, bias=True)  # decoder to patch


        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))


        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_st_masking(self, x, mask_ratio):
        """
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        x_masked_all = None
        mask_all = None
        ids_restore_all = None
        for data in range(D):
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x[:, :, data], dim=1, index=ids_keep)

            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0

            mask = torch.gather(mask, dim=1, index=ids_restore)

            if data == 0:
                x_masked_all = x_masked.unsqueeze(-1)
                mask_all = mask.unsqueeze(-1)
                ids_restore_all = ids_restore.unsqueeze(-1)
            else:
                x_masked_all = torch.cat((x_masked.unsqueeze(-1), x_masked_all), dim=-1)
                mask_all = torch.cat((mask_all, mask.unsqueeze(-1)), dim=-1)
                ids_restore_all = torch.cat((ids_restore_all, ids_restore.unsqueeze(-1)), dim=-1)
        return x_masked_all, mask_all, ids_restore_all

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        attn_qv = 0
        for blk in self.blocks:
            x, attn_qv = blk(x)
        x = self.norm(x)
        return x, attn_qv

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        for blk in self.decoder_blocks:
            x, attn_qv = blk(x)
        x = self.decoder_norm(x)

        return x

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        pred = pred.reshape(imgs.shape)

        return pred


def mask_vit(**kwargs):
    model = MaskViT(depth=1, num_heads=4,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
