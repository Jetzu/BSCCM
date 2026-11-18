"""N053: Cytoland VSCyto2D encoder (CZ Biohub, BSD-3; UNeXt2 encoder = ConvNeXtV2-tiny with a 2x2 stride-2, 1-channel stem) as a
timm convnextv2_tiny. Converted state dict (keys renamed dwconv->conv_dw, layernorm->norm, stem.conv2d->stem.0, stem.norm->stem.1)
from https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt (fine-tuned model; its
encoder was FCMAE-pretrained on phase). Differences to stock timm fixed here: 2x2 stride-2 stem, no depthwise-conv bias (zeros,
frozen), LayerNorm eps 1e-5. The 1-channel stem is adapted to `in_chans` by repeating its weights / in_chans."""
import torch, torch.nn as nn, timm

CKPT = '/hnvme/workspace/v181bb10-bsccm/models/cytoland/converted/vscyto2d_encoder_timm_convnextv2_tiny_s2stem.pt'


def build_encoder(in_chans=24, drop_path=0.1):
    m = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=0, in_chans=1, drop_path_rate=drop_path)
    m.stem[0] = nn.Conv2d(1, 96, kernel_size=2, stride=2)
    for name, mod in m.named_modules():   # VisCy: stem + block norms eps 1e-5, downsample norms keep 1e-6 (red-team N053 F5)
        if (isinstance(mod, nn.LayerNorm) or type(mod).__name__.startswith('LayerNorm')) and '.downsample.' not in f'.{name}.': mod.eps = 1e-5
    sd = torch.load(CKPT, map_location='cpu', weights_only=True)
    sd = sd.get('state_dict', sd)
    for k, v in m.state_dict().items():   # depthwise-conv biases do not exist in VisCy -> zeros
        if k.endswith('conv_dw.bias') and k not in sd: sd[k] = torch.zeros_like(v)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith('head')]
    if missing or unexpected: raise SystemExit(f'VSCyto2D load mismatch: missing {missing[:5]} unexpected {unexpected[:5]}')
    for n_, p in m.named_parameters():
        if n_.endswith('conv_dw.bias'): p.requires_grad_(False)
    w = m.stem[0].weight.data   # (96, 1, 2, 2) -> (96, in_chans, 2, 2), same response to a constant image
    stem = nn.Conv2d(in_chans, 96, kernel_size=2, stride=2); stem.weight.data = w.repeat(1, in_chans, 1, 1) / in_chans
    stem.bias.data = m.stem[0].bias.data.clone(); m.stem[0] = stem
    return m
