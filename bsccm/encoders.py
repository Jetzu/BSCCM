"""N053: pretrained ViT / ConvNeXt backbones for the 24-channel label-free input (23 LED log-normalised images + phase).

Adapter = learned per-LED gate (24 scalars, init 1; |gate| after training = learned LED weighting) -> resize to the backbone's
native token grid -> backbone with its RGB patch-embedding adapted to 24 input channels (timm adapt_input_conv: RGB weights
repeated and rescaled by 3/24, so a constant image gives the same response as a grey RGB image). Output = pooled feature vector.

archs:  dinov3_s     timm vit_small_patch16_dinov3.lvd1689m (21.6 M), 128 -> 256 px (16x16 tokens)
        dinov3_b     timm vit_base_patch16_dinov3.lvd1689m  (85.7 M), 128 -> 256 px
        dinobloom_b  DinoBloom-B (DINOv2 ViT-B/14 tuned on blood smears; 85.7 M), 128 -> 224 px (16x16 tokens)
        cytoland     VSCyto2D UNeXt2 encoder (ConvNeXtV2), see build(); weights path in CYTO_CKPT
Weights are read from local files only (compute nodes have no internet)."""
import glob, os, torch, torch.nn as nn, torch.nn.functional as F, timm
from timm.models._manipulate import adapt_input_conv

W = '/hnvme/workspace/v181bb10-bsccm'
HF = W + '/models/hf'
ARCHS = ('dinov3_s', 'dinov3_b', 'dinobloom_b', 'cytoland')


def _snapshot(repo):
    f = sorted(glob.glob(f'{HF}/models--timm--{repo}/snapshots/*/model.safetensors'))
    if not f: raise SystemExit(f'N053: no local weights for {repo} under {HF}')
    return f[-1]


class LEDGate(nn.Module):
    def __init__(s, c):
        super().__init__(); s.g = nn.Parameter(torch.ones(c))

    def forward(s, x): return x * s.g.view(1, -1, 1, 1).to(x.dtype)


class LFBackbone(nn.Module):
    def __init__(s, arch, in_chans=24, drop_path=0.1):
        super().__init__()
        if arch not in ARCHS: raise SystemExit(f'unknown N053 arch {arch}')
        s.arch, s.gate = arch, LEDGate(in_chans)
        if arch in ('dinov3_s', 'dinov3_b'):
            repo = {'dinov3_s': 'vit_small_patch16_dinov3.lvd1689m', 'dinov3_b': 'vit_base_patch16_dinov3.lvd1689m'}[arch]
            s.size = 256
            s.net = timm.create_model(repo.split('.')[0], pretrained=True, num_classes=0, in_chans=in_chans, img_size=s.size,
                                      drop_path_rate=drop_path, pretrained_cfg_overlay=dict(file=_snapshot(repo)))
        elif arch == 'dinobloom_b':
            s.size = 224
            s.net = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=0, img_size=224, in_chans=in_chans,
                                      drop_path_rate=drop_path)
            sd = torch.load(f'{W}/models/DinoBloom-B.pth', map_location='cpu', weights_only=True)['teacher']
            sd = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.') and k != 'backbone.mask_token'}
            sd['patch_embed.proj.weight'] = adapt_input_conv(in_chans, sd['patch_embed.proj.weight'])
            missing, unexpected = s.net.load_state_dict(sd, strict=False)
            if missing or unexpected: raise SystemExit(f'DinoBloom-B load mismatch: missing {missing} unexpected {unexpected}')
        else:   # cytoland: ConvNeXtV2 encoder of the VSCyto2D UNeXt2 (keys remapped to timm by bsccm.cytoland)
            from bsccm.cytoland import build_encoder
            s.size = 128; s.net = build_encoder(in_chans, drop_path)
        s.num_features = s.net.num_features

    def forward(s, x):
        x = s.gate(x)
        if x.shape[-1] != s.size: x = F.interpolate(x, size=(s.size, s.size), mode='bilinear', align_corners=False)
        return s.net(x)

    def led_weights(s):
        """|gate_c| x L2 norm of the patch-embedding (or stem) weights of input channel c: effective per-channel gain."""
        w = s.net.patch_embed.proj.weight if hasattr(s.net, 'patch_embed') else s.net.stem[0].weight
        return (s.gate.g.detach().abs() * w.detach().float().transpose(0, 1).flatten(1).norm(dim=1)).cpu().tolist()
