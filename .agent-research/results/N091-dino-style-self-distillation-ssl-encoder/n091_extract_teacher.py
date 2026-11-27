"""Frozen DINO TEACHER features for the N087 clustering rows. n087_extract.py verifies SSL exports against ckpt['model'] (the STUDENT);
DINO exports the teacher, so load_encoder is replaced by a loader that checks the teacher instead (and that the held-out draw was excluded)."""
import sys, torch, timm
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex'); import n087_extract as X
def load_teacher(kind, path, held, batch, ssl_excluded_batch=None):
    ck = torch.load(str(path).replace('encoder_final.pt', 'ckpt.pt'), map_location='cpu', weights_only=False)
    ex = ck['config']['exclude_batch']; assert ex == [0], ex   # b10 cluster rows are draw-0 cells: the encoder must not have seen draw 0
    sd = torch.load(path, map_location='cpu', weights_only=True)
    assert all(torch.equal(sd[k], ck['teacher'][k]) for k in sd), 'export is not the teacher'
    m = timm.create_model('resnet50', pretrained=False, num_classes=0, in_chans=48); m.load_state_dict(sd, strict=True); return m, m.num_features
X.load_encoder = load_teacher
X.extract(X.parse_args(sys.argv[1:]))
