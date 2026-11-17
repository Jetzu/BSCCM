"""Source-only raw-count nuisance augmentation (N041); no phase/fluorescence.

Counts remain float32 until GPUNorm. P=0 returns the original tensor without RNG
consumption. Active augmentation adds noise to the observed image; it never denoises
or replaces its existing artefacts. See report.md for calibration and geometry.
"""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from bsccm.data import LED

# BF includes the single coherent low-NA LED119; other groups follow CHSETS.
FAMILIES = (0,) + (2,) * 8 + (3,) * 9 + (1,) * 4 + (0,)
FAMILY_NAMES = ('BF', 'DPC', 'DF-low', 'DF-high')

# Mean ray slope / 0.166 um per pixel, (dx, dy) pixels per um of defocus.
# Waller-Lab/BSCCM quasi_dome_design.json, z_offset=8 mm, equal LED weights,
# n_sample=1.33 (PBS). Upstream illumination_to_led_indices defines the masks:
# DF_nn = NA in [nn/100, nn/100+.05], DPC = [.4,.5]; Left has NA_x>=0,
# Right NA_x<=0, Bottom NA_y<=0, Top NA_y>=0. Assume image x=NA_x, y=-NA_y.
# Symmetric full rings/BF have zero first moment. This is a centroid-shift
# approximation, not a wave-optics defocus model (no ring blur/phase update).
SHIFT_PX_PER_UM = (
    (0., 0.), (0., 0.), (0., 1.7065986542), (-1.7065986542, 0.),
    (0., 0.), (0., 0.), (0., 2.0244791700), (-2.0244791700, 0.),
    (0., 0.), (0., 0.), (0., 2.5050092235), (-2.5050092235, 0.),
    (0., 0.), (0., 0.), (0., 3.1475291164), (-3.1475291164, 0.),
    (0., 0.), (0., 0.), (0., 1.3703818677), (1.3703818677, 0.),
    (-1.3703818677, 0.), (0., -1.3703818677), (0., -1.8963940623),
)


def calibrate_source_borders(led, rows, batch, split, hist_row, *, max_rows=512,
                             seed=0, border=8):
    """Estimate an effective ADU/electron gain from source split-0 borders.

    Only validated `rows` are ever indexed in `led` (read-only mmap supported).
    Each non-overlapping border 2x2 tile supplies mean signal mu=mean(raw)-30
    and Haar residual h=(a-b-c+d)/2, which cancels constant/linear shading and
    has Var(h)=Var(pixel) for independent noise. Within 12 signal quantiles,
    median(h^2)/median(chi2_1) estimates variance robustly. A line fit with a
    nonnegative intercept models var=g*mu+read_var. Median positive channel
    slopes gives one camera gain. This spatial effective estimate is not a
    temporal photon-transfer calibration; PRNU/texture can inflate it.
    """
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    if not len(rows) or max_rows < 16:
        raise ValueError('need source calibration rows and max_rows >= 16')
    if (rows < 0).any() or (rows >= len(batch)).any():
        raise ValueError('invalid calibration row')
    draws = np.unique(np.asarray(batch)[rows])
    if len(draws) != 1 or (np.asarray(split)[rows] != 0).any() or (np.asarray(hist_row)[rows] >= 0).any():
        raise ValueError('calibration requires ONE source draw, split 0, hist_row < 0')
    rows = np.sort(np.random.default_rng(seed).permutation(rows)[:max_rows])
    if len(rows) < 16:
        raise ValueError('need at least 16 source split-0 calibration cells')
    h, w = led.shape[-2:]
    if len(led.shape) != 4 or led.shape[1] != len(LED) or h % 2 or w % 2 or not 2 <= border < min(h, w) // 2:
        raise ValueError('expected (N,23,H,W), even H/W, valid border width')
    yy, xx = np.mgrid[:h // 2, :w // 2] * 2
    mask = (yy < border) | (yy >= h - border) | (xx < border) | (xx >= w - border)
    means, residuals = [], []
    for row in rows:
        # Select only border tiles before conversion; the interior is not fitted.
        raw = led[int(row)]
        aa, bb = raw[:, 0::2, 0::2][:, mask].astype(np.float32), raw[:, 0::2, 1::2][:, mask].astype(np.float32)
        cc, dd = raw[:, 1::2, 0::2][:, mask].astype(np.float32), raw[:, 1::2, 1::2][:, mask].astype(np.float32)
        mu = (aa + bb + cc + dd) / 4 - 30
        hh = (aa - bb - cc + dd) / 2
        valid = (np.minimum.reduce([aa, bb, cc, dd]) > 30) & (np.maximum.reduce([aa, bb, cc, dd]) < 4095)
        means.append(np.where(valid, mu, np.nan))
        residuals.append(np.where(valid, hh * hh, np.nan))
    means, residuals = np.stack(means), np.stack(residuals)
    channel_fits, gains, offset_levels = [], [], []
    for c, name in enumerate(LED):
        mu, hs = means[:, c].ravel(), residuals[:, c].ravel()
        ok = np.isfinite(mu) & np.isfinite(hs)
        mu, hs = mu[ok], hs[ok]
        if len(mu) < 1000:
            raise ValueError(f'insufficient nonsaturated border samples: {name}')
        edges = np.unique(np.quantile(mu, np.linspace(0, 1, 13)))
        bins = np.searchsorted(edges[1:-1], mu, side='right')
        levels, variances = [], []
        for k in range(len(edges) - 1):
            take = bins == k
            if take.sum() >= 100:
                levels.append(float(np.median(mu[take])))
                variances.append(float(np.median(hs[take]) / 0.4549364231195727))
        levels, variances = np.asarray(levels), np.asarray(variances)
        if len(levels) < 3 or np.ptp(levels) < 10:
            raise ValueError(f'insufficient signal range for camera gain fit: {name}')
        slope, intercept = np.linalg.lstsq(np.column_stack([levels, np.ones_like(levels)]), variances, rcond=None)[0]
        # Boundary solution for nonnegative read variance, not an invented gain.
        if intercept < 0:
            slope, intercept = float(levels @ variances / (levels @ levels)), 0.
        if slope > 0 and np.isfinite(slope):
            gains.append(float(slope))
        # Small residual electronic baseline jitter: one noise SD of a
        # 128-pixel border mean, bounded to [0.5, 3] raw ADU below.
        offset_levels.append(float(np.sqrt(np.median(variances) / 128)))
        channel_fits.append(dict(channel=name, gain_adu_per_e=float(slope), read_var_adu2=float(intercept),
                                 signal_adu=levels.tolist(), variance_adu2=variances.tolist()))
    if len(gains) < len(LED) // 2:
        raise ValueError('camera gain fit failed on most channels; inspect source borders')
    gain = float(np.median(gains))
    gain_log_sd = None   # set by family_gain_log_sd() from GPUNorm-normalised borders (train_lf.py); PhysAug default .1 only for tests
    offset_sd = [float(np.clip(np.median([offset_levels[c] for c in range(len(LED)) if FAMILIES[c] == f]), .5, 3.))
                 for f in range(4)]
    return dict(version=2, source_draw=int(draws[0]), rows=rows.tolist(), seed=int(seed), border=int(border),
                rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
                gain_adu_per_e=gain, offset_sd_adu=offset_sd, family_names=list(FAMILY_NAMES),
                gain_log_sd=gain_log_sd,
                defocus_um=1., channel_fits=channel_fits,
                method='border 2x2 Haar; 12 signal bins; robust variance; nonnegative read intercept; median channel slope')


class PhysAug:
    """Batched GPU/CPU augmentation; phase is deliberately not an argument.

    Shared family gain/offset and one shared z per cell preserve LED identities.
    Centered Poisson draws g*Poisson(signal/g)-signal add variance g*signal
    without erasing the original sensor noise. One extra exposure's shot variance
    is added. Clip only perturbed samples to ADC [0,4095], without rounding.
    P<1 leaves unselected samples numerically exact.
    """
    def __init__(self, probability, gain_adu_per_e, offset_sd_adu, *, device='cpu',
                 gain_log_sd=.1, defocus_um=1., shot_scale=1.):
        gls = np.broadcast_to(np.asarray(gain_log_sd, dtype=np.float32), (4,)).copy()   # scalar or one log-sd per family
        if not np.isfinite(gls).all() or (gls < 0).any(): raise ValueError('gain_log_sd must be finite and nonnegative')
        vals = (probability, gain_adu_per_e, float(gls.max()), defocus_um, shot_scale)
        if not all(np.isfinite(v) for v in vals) or not 0 <= probability <= 1 or gain_adu_per_e <= 0 or min(vals[2:]) < 0:
            raise ValueError('invalid augmentation parameters')
        offsets = np.asarray(offset_sd_adu, dtype=np.float32)
        if offsets.shape != (4,) or not np.isfinite(offsets).all() or (offsets < 0).any():
            raise ValueError('offset_sd_adu must have four nonnegative finite values')
        self.probability, self.camera_gain = float(probability), float(gain_adu_per_e)
        self.gain_log_sd, self.defocus_um, self.shot_scale = torch.tensor(gls, device=device), defocus_um, shot_scale
        self.families = torch.tensor(FAMILIES, device=device, dtype=torch.long)
        self.offset_sd = torch.tensor(offsets, device=device)
        self.shift_per_um = torch.tensor(SHIFT_PX_PER_UM, device=device, dtype=torch.float32)
        self._grids = {}

    def sample_parameters(self, batch_size, *, generator=None):
        d = self.families.device
        active = torch.rand(batch_size, device=d, generator=generator) < self.probability
        gain = torch.exp(torch.randn(batch_size, 4, device=d, generator=generator) * self.gain_log_sd)
        offset = torch.randn(batch_size, 4, device=d, generator=generator) * self.offset_sd
        z = (2 * torch.rand(batch_size, device=d, generator=generator) - 1) * self.defocus_um
        return dict(active=active, gain=gain, offset=offset, z_um=z,
                    shift_xy=z[:, None, None] * self.shift_per_um[None])

    @torch.no_grad()
    def __call__(self, raw, *, generator=None):
        if self.probability == 0:
            return raw
        if raw.ndim != 4 or raw.shape[1] != len(LED) or raw.device != self.families.device:
            raise ValueError('expected raw (B,23,H,W) on the augmentation device')
        params = self.sample_parameters(len(raw), generator=generator)
        x = raw.float()
        if self.defocus_um > 0:
            h, w = raw.shape[-2:]
            key = (h, w)
            if key not in self._grids:
                yy, xx = torch.meshgrid((torch.arange(h, device=x.device) + .5) * (2 / h) - 1,
                                        (torch.arange(w, device=x.device) + .5) * (2 / w) - 1, indexing='ij')
                self._grids[key] = torch.stack([xx, yy], -1)[None]
            shift = params['shift_xy'].reshape(-1, 1, 1, 2)
            # Subtract displacement: positive dx moves image content right.
            grid = self._grids[key] - shift * x.new_tensor([2 / w, 2 / h])
            shifted = F.grid_sample(x.reshape(-1, 1, h, w), grid, mode='bilinear', padding_mode='border', align_corners=False)
            shifted = shifted.reshape_as(x)
            # Preserve zero-shift channels exactly (avoid interpolation roundoff).
            movable = self.shift_per_um.abs().sum(1) > 0
            x = torch.where(movable[None, :, None, None], shifted, x)
        signal = (x - 30) * params['gain'][:, self.families, None, None]
        out = 30 + signal
        if self.shot_scale > 0:
            g = self.camera_gain * self.shot_scale
            rate_signal = signal.clamp_min(0)
            out = out + g * torch.poisson(rate_signal / g, generator=generator) - rate_signal
        out = (out + params['offset'][:, self.families, None, None]).clamp(0, 4095)
        return torch.where(params['active'][:, None, None, None], out, raw.float())


def family_gain_log_sd(norm_border_dc, s_c, mult=2.):
    """N041 red-team M1: per-family brightness log-sd from the variation the MODEL actually sees. norm_border_dc (cells, 23) = mean
    border value of GPUNorm-normalised source split-0 crops (flat field + slide scale already divided out), s_c (23,) the per-channel
    log scale; robust sd (1.4826 MAD) across cells -> log units (x s_c) -> MIN over the family's channels (one gain is shared by the
    family, so no channel is perturbed beyond ~mult x its own observed variation) -> x mult (headroom for a new draw)."""
    d = np.asarray(norm_border_dc, dtype=np.float64)
    sd = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0) * np.asarray(s_c, dtype=np.float64)
    return [float(mult * np.min([sd[c] for c in range(len(LED)) if FAMILIES[c] == f])) for f in range(4)]
