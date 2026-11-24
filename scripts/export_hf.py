"""Export weights-only checkpoints of the released BSCCM models to a Hugging Face staging folder (N089).
For every fine-tuned run: weights.safetensors (model state dict, fp32), config.json (training config, epochs, final log entry, source
path), eval_preds.npz + result.json (held-out predictions / metrics, so ensembles can be scored without re-running).
For every SSL run: encoder.safetensors (+ projector.safetensors from ckpt.pt when present) and config.json.
Optimizer/scheduler states, row lists and RNG states are NOT exported.
usage: python3 -I export_hf.py STAGE_DIR [initial|n085]   (n085 = members added by the N085 E8 ensemble)"""
import sys, os, json, shutil, torch
from safetensors.torch import save_file
RUNS = '/hnvme/workspace/v181bb10-bsccm/runs'; OUT = sys.argv[1]; GROUP = sys.argv[2] if len(sys.argv) > 2 else 'initial'
FT = {'ensemble_N053/resnet50_full_view': 'n048_v9_base', 'ensemble_N053/resnet50_zoom2': 'n048_v9_zoom2', 'ensemble_N053/cytoland_convnextv2_tiny': 'n053_cytoland',
      'control/resnet50_ssl_v10': 'n051_v10_base',
      'composite_N063/twopart_fovre_seed2': 'n063_tpfov_s2', 'composite_N063/twopart_fovre_seed3': 'n063_tpfov_s3',
      'composite_N063/panel_seed2': 'n063_panel_s2', 'composite_N063/panel_seed3': 'n063_panel_s3'}
SPECIAL = {'ensemble_N084/convnext_s_member/draw1_to_draw0': 'n083_cnxs60_s1_b10', 'ensemble_N084/convnext_s_member/draw0_to_draw1': 'n084_cnxs120_s1_b01'}
SSL = {'ssl_encoders/resnet50_ssl_trained_on_draw1': 'n031_ssl_aug_ex0', 'ssl_encoders/resnet50_ssl_trained_on_draw0': 'n031_ssl_aug_ex1',
       'ssl_encoders/convnext_s_ssl60_trained_on_draw1': 'n078_ssl_cnxs_60_ex0', 'ssl_encoders/convnext_s_ssl120_trained_on_draw0': 'n083_ssl_cnxs_120_ex1'}
def clean(sd): return {k: v.detach().float().contiguous().clone() for k, v in sd.items() if torch.is_tensor(v)}
def ft(dst, run):
    s = torch.load(f'{RUNS}/{run}/ckpt.pt', map_location='cpu', weights_only=False); os.makedirs(dst, exist_ok=True)
    save_file(clean(s['m']), f'{dst}/weights.safetensors')
    json.dump(dict(source_run=run, epochs_done=s.get('ep'), train_config=s.get('cfg'), final_log=(s.get('log') or [None])[-1]),
              open(f'{dst}/config.json', 'w'), indent=1, default=str)
    for f in ('eval_preds.npz', 'result.json'):
        if os.path.exists(f'{RUNS}/{run}/{f}'): shutil.copy(f'{RUNS}/{run}/{f}', f'{dst}/{f}')
if GROUP == 'n085':   # N085 E8 members not in the initial release (composite z2/cyto from N065, ConvNeXt-S 120-ep plain + composite)
    FT = {'ensemble_N085_E8/composite_zoom2/twopart_fovre': 'n065_tpfov_z2', 'ensemble_N085_E8/composite_zoom2/panel': 'n065_panel_z2',
          'ensemble_N085_E8/composite_cytoland/twopart_fovre': 'n065_tpfov_cyto', 'ensemble_N085_E8/composite_cytoland/panel': 'n065_panel_cyto',
          'ensemble_N085_E8/convnext_s_composite/twopart_fovre': 'n085_cnxs120_tpfov_s0', 'ensemble_N085_E8/convnext_s_composite/panel': 'n085_cnxs120_panel_s0'}
    SPECIAL = {'ensemble_N085_E8/convnext_s_plain/seed0/draw1_to_draw0': 'n085_cnxs120_plain_s0_b10', 'ensemble_N085_E8/convnext_s_plain/seed1/draw1_to_draw0': 'n085_cnxs120_plain_s1_b10',
               'ensemble_N085_E8/convnext_s_plain/seed0/draw0_to_draw1': 'n083_cnxs120_s0_b01', 'ensemble_N085_E8/convnext_s_plain/seed1/draw0_to_draw1': 'n084_cnxs120_s1_b01'}
    SSL = {'ssl_encoders/convnext_s_ssl120_trained_on_draw1': 'n085_ssl_cnxs_120_ex0'}
for name, run in FT.items():
    for d, lab in (('b10', 'draw1_to_draw0'), ('b01', 'draw0_to_draw1')): ft(f'{OUT}/{name}/{lab}', f'{run}_{d}'); print('ok', name, d, flush=True)
for name, run in SPECIAL.items(): ft(f'{OUT}/{name}', run); print('ok', name, flush=True)
for name, run in SSL.items():
    dst = f'{OUT}/{name}'; os.makedirs(dst, exist_ok=True)
    save_file(clean(torch.load(f'{RUNS}/{run}/encoder_final.pt', map_location='cpu', weights_only=True)), f'{dst}/encoder.safetensors')
    cfg = dict(source_run=run)
    if os.path.exists(f'{RUNS}/{run}/ckpt.pt'):
        c = torch.load(f'{RUNS}/{run}/ckpt.pt', map_location='cpu', weights_only=False)
        if 'projector' in c: save_file(clean(c['projector']), f'{dst}/projector.safetensors')
        cfg.update(config=c.get('config'), epoch=c.get('epoch'))
    json.dump(cfg, open(f'{dst}/config.json', 'w'), indent=1, default=str); print('ok', name, flush=True)
