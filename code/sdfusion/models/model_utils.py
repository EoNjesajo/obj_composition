from termcolor import colored
import torch
import os

from sdfusion.models.networks.vqvae_networks.network import VQVAE


def load_vqvae(vq_conf, ckpt, opt=None):
    assert type(ckpt) == str
    
    # vq_ckpt = os.path.join(ckpt, "vqvae-snet-all.pth")

    # init vqvae for decoding shapes
    mparam = vq_conf.model.params
    n_embed = mparam.n_embed
    embed_dim = mparam.embed_dim
    ddconfig = mparam.ddconfig

    n_down = len(ddconfig.ch_mult) - 1

    vqvae = VQVAE(ddconfig, n_embed, embed_dim)
    
    map_fn = lambda storage, loc: storage
    state_dict = torch.load(ckpt, map_location=map_fn)
    if 'vqvae' in state_dict:
        vqvae.load_state_dict(state_dict['vqvae'])
    else:
        vqvae.load_state_dict(state_dict)

    print(colored('[*] VQVAE: weight successfully load from: %s' % ckpt, 'blue'))
    vqvae.requires_grad = False

    vqvae.to(opt.device)
    vqvae.eval()
    return vqvae