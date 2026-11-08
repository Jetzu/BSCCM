# BSCCM: paper + code deep-read (N003 agent report)

Date: 2026-11-08. Author: Claude subagent (paper/code reader). Scope: answer 9 data questions from the paper, the official repo, the Dryad page, and a few read-only checks on our local copy.

## Sources and how they were read

| Tag | Source | Notes |
|---|---|---|
| **[P]** | Pinkard, Liu, Nyatigo, Fletcher, Waller, "The Berkeley Single Cell Computational Microscopy (BSCCM) Dataset", arXiv:2402.06191 **v1** (9 Feb 2024). HTML: https://arxiv.org/html/2402.06191v1 ; PDF: https://arxiv.org/pdf/2402.06191 | Only v1 exists (`/html/...v2` and `v3` return 404, and the abs page lists only [v1]). Cited by section name (e.g. [P §S1.1 "Histology staining"]). The figure PNGs (Fig S2, S4, S5, Fig 3) were looked at directly (https://arxiv.org/html/2402.06191v1/data_processing.png, `cell_type_clustering.png`, `demixing_details.png`, `bsccm_led_array_patterns.png`). |
| **[R]** | https://github.com/Waller-Lab/BSCCM at commit `64de2e0f681a09be3f2a91ae4ead27628bab353d` (shallow clone, since deleted) | `.py` files are cited by line number. Notebooks are cited as `file.ipynb` JSON line N (the line in the raw .ipynb file) or "cell k" (0-based code/markdown cell index). |
| **[D]** | Dryad record https://doi.org/10.5061/dryad.sxksn038s (landing page read with WebFetch) | Versions: 2024-02-08 initial; 2025-05-06 "fixed a bug in the BSCCM-tiny metadata". The page lists no known issues and no donor/IRB text. |
| **[L]** | Read-only checks on our local copy (`/hnvme/workspace/v181bb10-bsccm/`: `bsccm_index.parquet`, `bsccm_sm.parquet`, `BSCCM.zarr.zip`, `pack/*.npy`) | Scripts are in this session's scratchpad (not the repo). The numbers below were measured first-hand. |
| **[G]** | Getting_started.ipynb in [R] | The official tutorial. |
| **[I]** | Pinkard et al., "Information-driven design of imaging systems", arXiv:2405.20559v4 (https://arxiv.org/html/2405.20559v4) | A downstream use of BSCCM by the same first author. |

Not reachable: H. Pinkard's PhD thesis "Data-Driven Information-Optimal Computational Microscopy" (UC Berkeley 2022, https://escholarship.org/uc/item/0kw136nx). [P §S1.1] calls the autofocus "described in full in chapter 2", so the paper was adapted from this thesis. eScholarship serves a bot check, so it was not read. It may hold donor and batch details that the paper leaves out **[unverified]**.

---

## Q1. Histology: acquisition, matching, mismatch rate, why only unstained, which stain

**Acquisition (same slide, stained after LED imaging):** yes.
- The imaging chamber was designed to "be disassembled so that a subsequent histology staining step could be applied, without moving cells" [P §S1.1 "Assembling imaging chambers"]. It is a poly-L-lysine coverslip with a 600 µm acrylic spacer, waxed to a slide. Cells were fixed to the coverslip with 4 rounds of 16% PFA exchange [P §S1.1 "Filling imaging chambers with cells"].
- "After imaging in fluorescence and LED-array modes, **some chambers** were disassembled for histology staining" [P §S1.1 "Histology staining"]. The chamber was opened under water and the coverslip released with a razor blade. The epoxy-sealed end was cut off with a diamond knife. The coverslip was stained 1 min in **Wright's stain (Sigma-Aldrich #45253)**, then dipped 5x in 25% ethanol phosphate buffer at pH 6.65 and 2x in low-concentration phosphate buffer at pH 6.65. It was blotted, dried overnight, scraped of wax, mounted in Neo-Mount on a **fresh slide**, and cleaned with methanol.
- Why a separate step: "having a histology stain on the cells during imaging with the LED array would alter their absorption/scattering properties, and could potentially alter the fluorescence signal as well" [P §S1.1 "Histology staining"].
- Imaging: Zeiss Axio Observer, **63x 1.4 NA oil** objective, the same monochrome Basler acA2440-75um camera. RGB was formed by lighting the **red, green and blue brightfield LEDs of the same LED array one after another** [P §3 ¶2; §S1.1 "Microscope and data collection"].
  - Autofocus was not the single-shot method. Focal stacks were taken at a subset of XY positions, and a running average interpolated the focus offset elsewhere [P §S1.1].
  - Global metadata [L, `BSCCM/BSCCM_global_metadata.json` extracted from the zip]: histology `pixel_size_um` 0.0534, image_shape 398x398, channel_names [Red, Green, Blue], wavelengths 630/515/460 nm, exposures R55/G60/B70 ms, camera offset 30, gain 0 dB (the LED/fluor gain is 4 dB).
  - **398 px x 0.0534 µm = 21.25 µm, which equals the LED crop's 128 px x 0.166 µm = 21.25 µm.** The two crops cover the same physical field of view, which matters for registration.
- Bit depth: histology is stored as uint16 holding 12-bit values per channel ("398x398 36-bit") [P Table 1]. `read_image(..., 'histology')` divides by 2^12 by default (`convert_histology_rgb32`) [R `bsccm/bsccm.py` L257-258]. The docstring says "10-bits per pixel" (L214-215), which contradicts the code; the 2^12 divisor and "36-bit" agree with each other.

**Stain: Wright's, not Giemsa.**
- The paper gives Wright's three times, with a catalog number: [P §3 ¶2], [P Fig S1e caption], and [P §S1.1 "Histology staining"] ("Wright's stain (Sigma-Aldrich # 45253)"). Sigma 45253 is "Wright's stain solution for microscopy" (eosin Y + polychromed methylene blue; https://www.sigmaaldrich.com/product/sigma/45253).
- The tutorial alone says "Histology, Which are **Giemsa** stained cells" [G JSON L260]. Treat that as a documentation slip. Both are Romanowsky-type stains, so they look similar, but the paper's protocol with a catalog number takes precedence.

**Registration / matching method** [P §S1.2, last two paragraphs; Fig S2i]:
1. Histology cells were detected with a difference-of-Gaussians blob finder. Candidates with a low summed gradient (out-of-focus or empty crops) were removed. "This was the only exclusion step needed".
2. **One global rigid transform per slide** (translation + rotation, no scale, no local warp). It was fitted by grid search on the GPU (JAX), minimising the sum over histology cells of the distance to the nearest LED/fluor cell, and "confirmed by looking at the visual alignment" (Fig S2i).
3. Matching is then nearest-neighbour, described as "one-to-one matching of histology cell candidates to LED array/fluorescence ones". The LED/fluor cells had already passed the CNN QC filter, so a histology crop only exists for cells that survived that filter.

**Expected mismatch rate:** **not reported anywhere** [P], [R], [D].
- No per-pair distance threshold, residual statistics, or match fraction is given. The code that did the matching is **not in the repo**: grep finds no histology alignment code; `bsccm/creation/` holds only crop/tiny/upload notebooks [R].
- Risk factors that follow from the method [inference]:
  - Cells can be lost or shifted during disassembly, staining, and remounting on a new slide.
  - A single rigid transform cannot absorb local stage nonlinearity over a full slide scan (~16 h, several mm).
  - Density-dependent nearest-neighbour ambiguity.
  - Different objectives and focus.
- Fig S2i shows qDPC detections without a histology partner, so cell loss certainly happens. The fraction is not quantified [unverified].
- **Recommendation:** measure it ourselves (e.g. nuclear-mask or phase-vs-absorbance correlation per pair, flagging low-similarity pairs). The physical FOVs match (21.25 µm), so a centred-crop similarity score is meaningful.

**Why only unstained cells?** The paper never says so explicitly. It only says "some chambers" were stained, and the Getting_started tutorial says "a small subset" [G JSON L260]. Our data shows all 15,712 histology-matched cells are `antibodies=='unstained'` (verified upstream by the caller and re-checked below). Plausible reasons [inference, not stated by authors]:
- The stain could alter or quench fluorescence. The authors note it "could potentially alter the fluorescence signal".
- Unstained slides were the expendable controls. The antibody slides were the more valuable fluorescence measurements and were not disassembled.

The consequence for us: **no cell has histology + surface-marker labels together** (unstained cells have autofluorescence only). Histology and marker targets can only be linked through the label-free images.

Histology availability by slide [L] (appendix A4):
- Only **two slides** were histology-stained, both unstained (no-antibody) slides: **(batch 0, rep 0): 10,015 of 16,338 cells matched (61%)** and **(batch 1, rep 1): 5,697 of 14,055 (41%)**. The other unstained slide, (b1, rep 0) with 32,674 cells, has none.
- So **39-59% of the LED cells on a stained slide got no histology partner**. That is the observable non-match rate.
  - **First-hand split by FOV [L]:**
    - On (b0, rep0), 484 of 3,258 FOVs (15%) have zero matches and hold 27% of the unmatched cells. The matched FOVs stop at stage y ≤ 46,150 µm while LED FOVs reach 49,282 µm.
    - On (b1, rep1), 1,372 of 3,334 FOVs (41%) have zero matches and hold 65% of the unmatched cells. Matched FOVs stop at y ≤ 40,630 µm while LED FOVs reach 50,062 µm.
    - **So a large part of the non-matching is incomplete histology scan coverage (a missing slide region)**, especially in batch 1.
    - **Within FOVs that do have matches, only 68% (b0) / 66% (b1) of cells are matched.** That ~1/3 per-cell loss is from cell detachment during staining, the gradient focus filter, and/or matching failure.
    - The match rate does not depend on in-FOV radius (0.61-0.62 b0, 0.40-0.42 b1 across radius bins).
  - The rate of *wrong* matches among the 15,712 pairs is unknown. A ~1/3 per-cell loss in covered FOVs means nearest-neighbour matching sometimes had missing partners. **A wrong-neighbour match is plausible wherever a cell's true partner was lost but a neighbour lies close by** [inference], so a per-pair similarity QC is advisable.
- The b1 histology slide is also the one imaged **without the camera ROI** [P §S1.5], so its FOV geometry differs from all other slides.

## Q2. What is the `dpc` array?

**Quantitative phase (qDPC) reconstructed from the 4 half-annulus DPC images. It is not a raw DPC difference image.**
- [P §S2 "BSCCM_images.zarr"]: `"dpc": Quantitative differential phase contrast images (which were computed from raw DPC led-array images)`.
- [P §S1.2]: "the 4 differential phase contrast (DPC) images at each field of view were used as input to an inverse algorithm that computed quantitative phase [23] (qDPC)". Ref [23] is Tian & Waller, "Quantitative differential phase contrast imaging in an LED array microscope", Opt. Express 23:11394 (2015). That algorithm is a weak-object transfer function (WOTF) model with an L2/Tikhonov-regularised closed-form deconvolution.
- **It was computed per full field of view (2056x2056) and then cropped**: "qDPC images were calculated for each field of view" [P Fig S2c caption]. It was also used for blob detection and as one input to the QC CNN [P §S1.2]. A crop of a full-FOV reconstruction differs from a reconstruction of the 128x128 crop (boundary/low-frequency behaviour).
- dtype float32 [G cell 17; L zarr meta `dpc` `<f4`, shape (1,128,128)].
- The repo's qDPC implementation is `bsccm/phase/functional_dpc.py`: `generate_wotf` L52-75 and `tikhinov_solver` L91-109 with defaults `reg_u=1e-1, reg_p=5e-3`, returning absorption + 1j*phase.
- The tutorial recipe [G cell 57, JSON L1786-1808] is:
  - NA 0.5, `annulus_inner_NA` 0.40, rotations [180, 0, 90, 270] for [Bottom, Top, Left, Right], wavelength 0.515 µm, pixel 0.166 µm;
  - inputs `(raw - 30) / (background_50pct - 30)` (offset 30 = camera offset);
  - `phase = tikhinov_solver(...).imag[0]`.
- **The exact parameters used to make the released `dpc` arrays (regularisation, flat-field, sign convention, units) are not documented** [unverified]. By construction of the WOTF model the output is phase in radians (relative, without the DC/zero-frequency component) [inference].
- Code quirk: `annular_sources()` builds specs with key `'NAinner'` (`functional_dpc.py` L112), but `generate_source()` reads `'NA inner'` (L40). The inner NA is therefore silently ignored and the tutorial uses full half-discs, not half-annuli. This only affects users re-running the tutorial reconstruction, not the released `dpc`.
- **First-hand check [L] (appendix A5):** my own independent WOTF + Tikhonov qDPC reconstruction from the 4 stored DPC channels correlates **r = 0.984 median (min 0.970, n = 40 cells)** with the stored `dpc`. Naive DPC differences correlate ≈ 0. The stored values are ~2 (rad) at the cell centre and ~0 in the background, nearly identical in both batches.
- LED patterns [P Fig 3; R `bsccm/led_array_calibration.py` L128-166]: `DPC_*` = LEDs with 0.4 ≤ NA ≤ 0.5 restricted to one half-plane (Top/Bottom/Left/Right). Exposure is 17 ms each [L global metadata].

## Q3. Backgrounds zarr, `*_shading*` columns, recommended normalisation

### 3a. `BSCCM_backgrounds.zarr` (LED-array flat fields)
- Definition: "background images of each LED illumination pattern, computed by taking the pixel-wise median (smaller percentiles are available as well) across many fields-of-view, which effectively removes the contribution of the cells themselves and captures the microscope's illumination pattern" [P §3.2].
- "...over **200 images**, with possible options of 5, 10, 20, 40, and 50 (median)" percentiles [P §S2 "BSCCM_backgrounds.zarr"].
- Actual structure [L, read from the zip's central directory and `.zarray`]:
  - `BSCCM_backgrounds.zarr/<LED channel>/<p>_percentile`, p ∈ {5, 10, 20, 40, 50};
  - each array is **float64, shape (2056, 2056) = the full camera FOV**, chunks (257, 514).
  - There is **one array per LED channel**, including `LED119`, and it is **not per slide or per batch**: the group has no slide/batch level.
- There are **no fluorescence or DPC backgrounds** in this zarr (only LED channel names appear).
- The docstring of `get_background` says "specific to each physical slide" [R `bsccm.py` L344-351], but the code indexes only `[channel]['{p}_percentile']` and crops 128x128 around the cell's `position_in_fov_{x,y}_pix` (L358-362). **So it is one global flat field per channel; the "per slide" docstring is wrong** (verified from the zarr layout).
- It is absent from the tiny and MNIST variants (warning in `bsccm.py` L352-353; GitHub issue #6).
- Edge case: crops for cells within 64 px of the FOV border are truncated by the slicing in L361 (no padding). Cell positions span x 52-1967 and y 180-1916 [L], so a few cells near x < 64 get a background crop narrower than 128 px.

**Recommended LED normalisation (the authors' own recipe):**
- Shading correction is "raw measurements ... divided by an image taken on the same system under the same illumination, but without a sample" [P §2.1].
- The concrete form in the tutorial is `(raw - offset) / (background_p50 - offset)` with `offset = 30` [G cell 57, JSON L1786, L1804-1805]. The camera offset is 30 for LED, fluor and histology in global metadata [L].
- Caveat [P §S1.5, last bullet]: "From slide-to-slide and batch-to-batch, there are variations in the average intensity of images in a given channel ... the LED119 channel ... has very different average brightness between batch 0 and batch 1."
  - A single global flat field cannot remove per-slide gain differences.
  - Add a per-slide (batch x antibodies x slide_replicate) per-channel intensity normalisation (e.g. divide by the slide's median of background-region pixels), or at least evaluate across batches [recommendation].
- Exposures differ by channel: Brightfield 8 ms; DF_50 29 ms ... DF_90 510 ms; half-ring DF patterns use 2x exposure; DPC 17 ms; LED119 200 ms [L global metadata `led_array.exposure_ms`]. They were chosen "to effectively [use] the camera's dynamic range" [P §S1.1 "Exposures"], so raw counts are **not** comparable across channels without dividing by exposure (and LED count).

### 3b. Fluorescence columns (surface-marker CSV)
The pipeline is [R `fluorescence_processing/README.md`; `compute_fluorescence.ipynb`; `shading_correction.ipynb`; P §S1.3, Fig S3]:

| Column | Definition (code) |
|---|---|
| `Fluor_<ch>_total_raw` | **Mean** pixel value inside the circle inscribed in the 128x128 crop (r = 64 px) [R `compute_fluorescence.ipynb` JSON L74-78]. The paper says "summing"; the code uses `np.mean`. No offset subtraction. |
| `Fluor_<ch>_background` | Mean pixel value **outside** that circle (the 4 corners) = local background [JSON L80]. |
| `Fluor_<ch>_background_lowess` | Per slide (batch x antibodies x slide_replicate), a 2-D LOWESS (local linear, tricubic weights, alpha = 1000/n_cells, i.e. ~1000 nearest cells) of `_background` vs FOV position, evaluated at each cell [R `shading_correction.ipynb` cell 11, JSON L841; `shading_correction_util.py` L16-46]. |
| `Fluor_<ch>_shading` | Multiplicative field-of-view shading (vignetting) estimate, pooled over **all stained slides of both batches** (unstained excluded: "their fluorescence is low", JSON L933). It takes `total_raw - background_lowess` of the **spatially brightest 10%** of cells (per 20x20 spatial bin, > 90th percentile, JSON L961), caps at 2000 cells per slide, and fits a 2-D LOWESS evaluated at each cell [cell 20]. It is stored un-normalised. |
| `Fluor_<ch>_shading_corrected` | `(total_raw - background_lowess) / (shading / mean(shading))` [JSON L1283-1289]. **This is the background-subtracted, vignetting-corrected per-cell fluorescence** that feeds the unmixing. |

- The tutorial accessor `get_corrected_fluor()` returns exactly the 6 `_shading_corrected` columns [R `bsccm.py` L301-313]. Authors' recommendation: use `_shading_corrected` for raw-channel fluorescence and the `*_unmixed` columns for protein abundance [G cells 33-36, 41].
- **Possible code bug (affects `_shading` for 5 of 6 channels)** [R `shading_correction.ipynb` cell 20, JSON L1264-1265]:
  - The shading LOWESS uses values from `data_by_channel[channel_names[i]]` but positions from `data_by_channel[channel]`.
  - `channel` is a stale loop variable left from the previous cell (cells 16/18 loop `for c_index, channel in enumerate(channel_names)`, JSON L1065/L1090/L1164), so it equals `'Fluor_690-'`.
  - If the released CSV was produced by this exact code, the shading maps for the 426-446 ... 627-673 channels were fitted with **positions of different cells** and would be distorted.
  - The saved notebook output shows this cell was interrupted (`KeyboardInterrupt`), so we cannot tell which run produced the release.
  - **Data-side test [L] (appendix A3):**
    - The released `_shading` maps are weak (CV 3-10%) and rise 3-21% toward the FOV edge.
    - They are **anti-correlated (−0.39 to −0.81) with the empirical bright-cell spatial profile in 5 channels**.
    - Dividing by them makes marker-positive populations about 5-8% dimmer at r > 750 px instead of flatter.
    - The correction is therefore ineffective and slightly harmful, though small. The cause is [unverified].
  - Recommendation: use `total_raw − background_lowess` or refit the shading ourselves, if FOV-position effects matter.
- The single slide "unstained, batch 1, replicate 1" was imaged **without the camera ROI set** (all other slides used the central 2056x2056) [P §S1.5 bullet 3]. The code comment says this slide "has the fieldstop positioned slightly differently and so is not to be trusted" (`shading_correction.ipynb` JSON ~L935, cell 14). Its positions and shading are therefore suspect.

## Q4. Spectral unmixing: models, units, which column to use, crosstalk

**Model** [P §S1.3 "Fluorescence unmixing with non-negative matrix factorization", Fig S3e; R `demixing_util.py` L179-253; `do_demixing.ipynb`]:
- Objective: `X*, B* = argmin 1/(NC)||Y − (XS + B)||_F^2 + λ/(NM)||X diag(w)||_1 + β/(NC)||B||_1`, with X ≥ 0 and B ≥ 0.
- The **spectra S are fixed** (measured), so this is non-negative least squares with L1, solved by projected gradient descent with momentum (lr 1e3, momentum 0.9), stopping when the relative loss change over 50 iterations is < 5e-4 [R `demixing_util.py` L220-249].
- λ = 0.7 (`l1_reg = 7e-1`, `do_demixing.ipynb` JSON L165/L224) and β = 5e-2 (`background_reg` default, `demixing_util.py` L183; the code uses an L2 on B, not L1). The paper prints "λ = 7e-1, λ = 5e-2", a typo for β.
- Input Y = the 6 `_shading_corrected` columns, globally shifted so each column's minimum is 0.01 (`mixed_data -= (min − 1e-2)`, JSON L88).
- **Spectra** are measured **per batch**: mean(manually gated positive cells) − mean(other cells on the same single-antibody slide), clipped ≥ 0 and normalised to unit L2 norm [R `demixing_util.py` L20-63]. The gates are the `selection_example_<marker>_positive_cells_batch_<b>` columns.
  - The autofluorescence spectrum comes from the brightest-autofluorescence gate on the **unstained** slide [P §S1.3].
  - Gate sizes [L]: e.g. CD56 b0 191 cells, CD3 b0 872, CD16 b1 15,868. All gates sit on slide_replicate 0.
- **Fit grouping**: one fit per (antibodies x batch), pooling slide replicates, with a single background row B shared within that group [`do_demixing.ipynb` cells 3, 5]. The paper says B is per slide; the code shares B per condition x batch.

**Two models:**
1. **Single-antibody ("two-spectrum") model** → `<marker>_single_antibody_model_unmixed` + `autofluor_single_antibody_model_unmixed`.
   - Spectra = [that antibody's spectrum, autofluor]. Unstained cells get the autofluor spectrum only. **'all' cells are skipped**: "2 spectra model is misspecified" (`do_demixing.ipynb` JSON L153).
   - L1 weights: marker weight = (autofluor_unit · marker_unit)/marker_brightness, autofluor weight = 1 (JSON L163).
   - Data [L]: each `<M>_single_antibody_model_unmixed` is non-NaN **only** for cells with `antibodies == M` (100%) and NaN everywhere else. `autofluor_single...` is filled for all 8 single-antibody conditions + unstained and is NaN for 'all'.
2. **Full ("four-spectrum") model** → `CD123/HLA-DR/CD14_full_model_unmixed`, `CD3/CD19/CD56_full_model_unmixed`, `CD45_full_model_unmixed`, `CD16_full_model_unmixed`.
   - Pooled spectra:
     - CD3, CD19 and CD56 all use **the same fluorophore BV510**, so they are inseparable [P §S1.1 antibody list; Fig 5].
     - CD123 (BV421), HLA-DR (BV570) and CD14 (BV650) are pooled because their spectra are too similar under single 405 nm excitation [P §S1.3].
   - **The autofluorescence spectrum is excluded** (`unmixed_channel_names[:-1]`, JSON L221).
   - L1 weights come from the projection on the first right singular vector of the brightness-scaled spectra.
   - It was run on **every** condition (all, unstained, each single antibody) "for comparison purposes" [P §S1.3]. Data [L]: all 4 columns are non-NaN for 100% of cells.
- **Post-processing floor (important for targets):** per batch and per column, every value below the 0.5th percentile of the non-zero values, including exact zeros, is replaced by that percentile (`do_demixing.ipynb` JSON L188/L275, "makes it easier to deal with in log space"). A column's minimum therefore marks "zero / below detection", not a measured value.
  - Floor mass [L] over all conditions: CD123/HLA-DR/CD14 full: 64% (b0) / 51% (b1) of cells at the floor. CD45 full: 21% / 24%. CD3_single b1: 43%.
  - Within 'all' cells (A2): CD123/HLA-DR/CD14 59% (b0) / 55% (b1); CD16 31% / 8%; CD45 10% / 18%; CD3/CD19/CD56 0% / 9%.

**Units:** arbitrary.
- The tutorial labels axes "(Arbitrary units)" [G cell 43/45], and the paper says "relative abundance" [P §3.1].
- Concretely, X is the coefficient of a **unit-norm** spectrum fitted to background- and shading-corrected **mean-pixel camera counts** (offset not subtracted, but absorbed by B). So X scales with the fluorophore brightness/exposure and with the batch-specific spectrum normalisation.
- **Values are not comparable across batches** [inference + L]. Examples:
  - CD45_single median: b0 75.3 vs b1 37.8; p99 624 vs 111.
  - CD3_single median: b0 16.8 vs b1 1.5.
  - CD16_single median: b0 53 vs b1 137.
- They are also not comparable between the two models, or between the single-antibody model of marker A and marker B.

**Which column to use as the per-marker abundance target:**
- **(a) Single-antibody cells (`antibodies == M`):** `M_single_antibody_model_unmixed`. The authors say this is the more accurate one: "Though two-spectrum results are likely more accurate when available" [P §S1.3, last paragraph]. The tutorial calls it "the more accurate, two-spectrum model" [G cell 42].
  - Use it per batch (z-score or rank within batch x marker) and treat the floor value as censored/zero.
  - Optionally use `autofluor_single_antibody_model_unmixed` as an auxiliary target; it is available for all non-'all' cells.
- **(b) 'all' cells:** only the 4 pooled `*_full_model_unmixed` columns exist. This gives 4 targets (CD45, CD16, CD3/CD19/CD56 pooled = lymphocyte lineage, CD123/HLA-DR/CD14 pooled) and **no per-marker CD3 vs CD19 vs CD56 or CD14 vs HLA-DR vs CD123**.
  - The authors themselves consider CD123/HLA-DR/CD14 "too noisy to be useful" for labelling [P §S1.4]. 51-64% of all cells sit at its floor.
  - For batch 1 'all' cells, CD19 antibody was at **35% of the normal amount** [P §S1.5 bullet 1], so the CD3/CD19/CD56 pool is biased by batch there.

**Known crosstalk** [P §S1.3 last paragraphs; Fig S4b, inspected]:
- In the 4-spectrum model, "CD123, CD45 and CD16 all gave rise to signal in the CD3/CD19/CD56 spectra, which may be due in part to the fact that this model didn't explicitly account for autofluorescence".
  - Fig S4b shows CD45-only and CD16-only cells spreading to roughly 300-400 a.u. on the CD3/CD19/CD56 axis.
  - CD16-only cells also spread on the CD123/HLA-DR/CD14 axis (up to about 200+).
  - CD3-only cells barely register non-CD3 signal. CD45 vs CD16 crosstalk is small.
- Practical consequences:
  - The `CD3/CD19/CD56_full` column has a large autofluorescence-driven baseline in every condition. **Unstained cells, with no antibody, have a median of 74.5 (b0) / 63.0 (b1)**, against 87.4 / 70.6 for 'all' cells [L, A2]. Unstained cells also read about 35 on CD16_full and 6-9 on CD45_full.
  - Do not read a nonzero full-model value as marker presence. Calibrate against the same batch's unstained distribution.
  - Crosstalk is batch-dependent: CD45-only and CD16-only cells read ~10 a.u. on CD123/HLA-DR/CD14 in b1 but sit at the floor in b0 [L, A2].
  - CD123 has low brightness and poor separation (Fig S4b row 1), even with the 2-spectrum model.
- The design is fundamentally underdetermined: 6 channels, single 405 nm excitation, 6 fluorophores + autofluorescence [P §S1.1 "Exposures"; §S1.5 bullet 4, "hindsight is 20/20"].
- Exposure ratios were tuned to minimise the condition number of the 6x6 mixing matrix [P §S1.1 "Exposures"]. Fluor exposures: 426-446 600 ms, 500-550 1000, 550-570 1200, 585-625 1500, 627-673 2000, 690- 1600 [L global metadata].

## Q5. Batches and donors

- **Batch = a separate cell isolation (blood draw → RBC lysis → staining) on a different date**:
  - "The batch index captures the two biological replicates (i.e. distinct cell isolations) on two different dates. There is presumably some degree of biological variation between these two isolations, in addition to variation in antibody staining/processing/etc." [P §S2 "BSCCM_images.zarr"];
  - "Describes a batch of cells that were isolated in a single experiment. So cells with the same batch come from the same underlying population. However, they are split over many physical microscope slides, stained with different antibodies, and imaged on different dates." [G JSON L760].
- **The number of donors, and whether batch 0 and batch 1 are the same person, is never stated** [P], [R], [D].
  - The protocol says "12 mL of blood were drawn by venipuncture" [P §S1.1 "Cell isolation and staining"] without donor identity, count, health status, sex/age, IRB or consent.
  - "Biological replicates" is ambiguous, so treat it as **unknown: 1 or 2 donors** [unverified]. It may be in the thesis (not accessible).
  - Practical implication: batch confounds donor x isolation x staining x imaging session. With at most 2 donors, no donor-level generalisation claim is possible. **Batch is the strongest available domain shift: use it as the held-out split.**
- Batch-specific technical differences that are documented:
  - CD19 at 35% in the batch-1 'all' stain [P §S1.5];
  - LED119 brightness differs between batches [P §S1.5];
  - unmixing spectra are computed per batch [R `do_demixing.ipynb` cells 3, 5];
  - large scale differences in the unmixed values (Q4).
- Slide layout: "some conditions ... were split among multiple physical microscope slides and imaged on different dates. The vast majority of cells/slides did not have a second replicate" [P §S2]. The chambers were imaged "sequentially ... over the next 2 weeks" after isolation [P §S1.1 "Filling imaging chambers with cells"].
  - However, `imaging_date` [L, A4] spans **17 days in batch 0 (8/4 to 8/21/2018) and 25 days in batch 1 (8/27 to 9/21/2018)**. So fixed cells were stored for up to ~3.5 weeks, and **storage time is confounded with antibody condition** (one condition per slide per date).
  - The two isolations were roughly 3 weeks apart (first imaging dates 8/4 vs 8/27/2018) [inference from dates; isolation dates are not recorded].
- BSCCM-coherent is batch 1 only, with 'all' and 'unstained' conditions [P §S2; P Table 2].

Per-slide counts, histology and imaging dates [L]: see the appendix table.

## Q6. What the `selection_gated_*` gates mean; 3-class labels

**Provenance:**
- Labels were made by manually drawing polygon gates on **batch-0, 'all'-antibody, slide_replicate-0** cells only. They were drawn in the interactive `ScatterSelectorGating` GUI [R `gating_util.py` L32-170; gates saved as `selection_<name>`, L156-168] on log-scale scatter plots of the 4-spectrum unmixed values [R `marker_gating.ipynb` cell 2].
- "Cells were selected based on their expression of CD45, CD16, and CD3/CD19/CD56. The CD123/CD14/HLA-DR channel was deemed too noisy be useful" [P §S1.4].
- "It is unclear if all of these 10 classes are related to biological variability, rather than noise in the measurement procedure" [P §S1.4].
- Coverage [L]: **10,150 of the 10,210 batch-0 'all' cells (99.4%) are in at least one gate; 13 are in two.** No batch-1 or other-condition cell is gated.

**Index → gate → official class** (the label index is the CSV column order; the order was verified in [L] and matches `Make_TINY_and_MNIST.ipynb` JSON L161-170 and `get_cell_type_classification_data` in `bsccm.py` L364-405):

| idx | column | n [L] | official 3-class (bsccm.py L371-375, clustering.ipynb cell 9) | Fig S5 group | likely biology [inference, not authors'] |
|---|---|---|---|---|---|
| 0 | `selection_gated_CD3_hi_CD_16_lo` | 3037 | **Lymphocyte (0)** | Lymphocytes | T + B + NK (CD56) cells pooled. CD3/CD19/CD56 share BV510, so they **cannot be split into T/B/NK** in 'all' cells |
| 1 | `selection_gated_CD16_hi_CD45_mid` | 3704 | **Granulocyte (1)** | Granulocytes | Neutrophils (CD16-bright, CD45-dim/intermediate) |
| 2 | `selection_gated_CD16_hi_CD45_lomid` | 400 | Granulocyte (1) | Granulocytes | Neutrophils, lower CD45 |
| 3 | `selection_gated_CD16_hi_CD45_lo` | 803 | Granulocyte (1) | Granulocytes | Neutrophils with very low or failed CD45 signal. Possibly damaged or apoptotic cells [speculative] |
| 4 | `selection_gated_CD16_lo_CD45_lo` | 194 | excluded (-1) | **Red blood cells** | RBCs that survived lysis (CD45−, CD16−) |
| 5 | `selection_gated_CD16_none_CD45_mid_and_hi` | 419 | excluded (-1) | **Unknown** | CD45+ CD16− non-lymphoid. Could hold **eosinophils** (CD16−) and/or classical monocytes; the authors leave it unknown |
| 6 | `selection_gated_CD45_mid_CD_16_mid` | 97 | excluded (-1) | **Unknown** | unclear |
| 7 | `selection_gated_CD45_hi_CD_16_lo` | 104 | **Monocyte (2)** | Monocytes | classical (CD14+CD16−) monocytes? |
| 8 | `selection_gated_CD45_hi_CD_16_mid` | 1225 | Monocyte (2) | Monocytes | intermediate monocytes? |
| 9 | `selection_gated_CD45_hi_CD_16_hi` | 180 | Monocyte (2) | Monocytes | non-classical monocytes; possibly CD16+CD56+ **NK cells** too (they sit at moderate CD3/CD19/CD56 in Fig S5) [speculative] |

- **3-class labels exist only implicitly**: `BSCCM.get_cell_type_classification_data(ten_class_version=False)` maps 0→Lymphocyte(0), 1/2/3→Granulocyte(1), 7/8/9→Monocyte(2), and drops 4/5/6 (RBC, Unknown) [R `bsccm.py` L364-405]. There is no 3-class column in the CSV. Only batch 0 'all' cells are labelled ("These classification labels exist only for Batch 0, antibody condition 'all'", L377). The tutorial plots them as Lymphocyte / Granulocyte / Monocyte [G cell 19].
- Important gotchas:
  - The function relies on **CSV column order** (`[c for c in columns if 'selection_gated' in c]`). Reordering the columns silently permutes the labels. Hard-code the names.
  - The 10-class version concatenates gates without checking overlap. **13 cells are in two gates** [L], so they appear twice with different labels. De-duplicate them or drop them.
  - **`selection_lymphocytes`** (n = 3025, batch 0 'all' replicate 0 only) is **not referenced anywhere in the code or paper**. It is an extra, earlier lymphocyte gate: **2,852 of its cells overlap gate 0** (3,037). It is not part of the official labels.
  - **No eosinophil, basophil, or T/B/NK labels exist.** The panel has no eosinophil marker. Single-antibody slides (CD3-only, CD19-only, CD56-only, CD14-only, ...) do support per-slide marker-positive vs marker-negative labels.
  - The only marker gates on those slides are the `selection_example_*_positive_cells_batch_*` columns, and they were drawn as **spectrum-estimation examples**, not exhaustive positives. Do not use them as complete labels.

## Q7. `LED119` and BSCCM-coherent

- `LED119` = **a single LED (1-indexed LED #119) at NA (x, y) = (0.00, 0.40)**, i.e. an off-axis brightfield LED inside the 0.5 NA objective pupil [G cell 52: "The x and y NA of LED 119 are 0.00, 0.40"; R `led_array_calibration.py` L132-133; Fig 3 shows it as a single LED].
  - It is the one spatially coherent (plane-wave) channel in the default dataset: "22 images ... multi-LED pattern (brightfield, darkfield, or differential phase contrast) and a single image for each cell taken under illumination with a single LED" [P §3].
  - Exposure is 200 ms (vs 8 ms Brightfield) [L].
  - Intended for single-angle coherent techniques and as a coherent-vs-incoherent illumination comparison. [I] compares Brightfield vs DPC vs "Single LED" illumination for protein-expression information content and notes the single-LED channel's "sensitivity to optical imperfections".
  - **Known issues:**
    - A large batch-0 vs batch-1 mean brightness difference [P §S1.5 last bullet].
    - **Strong fixed-pattern coherent artifacts in its FOV background** [L, Q8 item 2]: fringe rings, dark disks, and a dark smear in the bottom-right corner, with local contrast about ±50% vs ±3-5% for the multi-LED patterns.
    - Normalise it per slide or batch and flat-field it with its background, or exclude it from models meant to cross batches.
- **BSCCM-coherent** (4,304 cells, 30 GB, batch 1, 'all' + 'unstained' only, no histology) has 566 channels, each a single LED [P §3 "BSCCM-coherent"; Table 1-2].
  - Purpose: "Techniques such as diffraction tomography and Fourier ptychography often rely on measurements taken with spatially coherent plane wave illumination from different angles" [P §3].
  - The 581-LED quasi-dome has its low-NA LEDs affected by an oil drop inside the objective. "When imaging with certain low-NA single LEDs, this created a strongly visible artifact across the field of view. Because of this, these LEDs are excluded from BSCCM-coherent" [P §S1.5 bullet 2]. This explains 566 < 581.
  - The coherent set's unmixing reused BSCCM's batch spectra (`do_demixing.ipynb` cell 1: `bsccm_with_spectra` = BSCCM, `bsccm_with_data` = coherent).
  - **Not relevant to our 23-channel BSCCM training unless we want angle-resolved data.**

## Q8. Known data issues and recommended exclusions

Documented by the authors [P §S1.5 "Known imperfections"] unless noted otherwise:
1. **CD19 at 35%** in batch-1 'all' cells. This biases `CD3/CD19/CD56_full` in batch 1 'all'.
2. **Oil drop on an internal objective lens element.** It produces a "strongly visible artifact across the FOV" for certain low-NA single LEDs. Those LEDs were dropped from BSCCM-coherent. In the 23-channel BSCCM, `LED119` (NA 0.40) is the single-LED channel.
   - **First-hand [L]:** I decoded the 50th-percentile backgrounds (2056², float64) from the zip.
   - **`LED119`'s background has strong fixed-pattern coherent artifacts.** It shows Airy-ring/fringe patterns from dust or defects, several dark disks, and a large dark smear at the bottom-right of the FOV (x ≈ 1500-2050, y ≈ 1800-2050).
     - Values span 83.5-686.5 (p1/median/p99 = 170/394/477), i.e. local contrast of about ±50%.
     - By comparison: Brightfield p1/med/p99 = 2966/3054/3143 (±3%); DF_50 232/244/254 (±5%); DPC_Left 1758/1948/2041 (smooth, darker right edge).
   - Whether this is the oil drop itself or dust is not determinable [unverified].
   - **Recommendation:** treat `LED119` as artifact-laden and position-dependent. At minimum, divide by its own background (`(raw−30)/(bg−30)`; coherent artifacts are only approximately multiplicative). Better, exclude it, or exclude cells in the artifact region, for models that must generalise. The multi-LED patterns look clean (smooth illumination only).
3. **The "unstained / batch 1 / slide_replicate 1" slide was imaged without the camera ROI.** This may change fluorescence exposure and spatial variation, and the code comment calls its fieldstop position "not to be trusted" [R `shading_correction.ipynb` cell 14]. **Recommended:** exclude it from fluorescence-derived targets, and check its FOV positions and LED crops.
   - **First-hand [L]:** its in-FOV cell positions are shifted. x spans 52-1771 vs 211-1967 on the other slides (y 211-1907 vs ~180-1911). That is consistent with a different camera ROI offset of ~160-200 px in x.
   - The global `BSCCM_backgrounds.zarr` flat fields (one per channel, Q3a) are therefore likely **mis-registered by ~160-200 px for this slide** [inference].
   - **This is also the batch-1 histology slide (5,697 of the 15,712 histology pairs).** Use a slide-specific flat field (e.g. a per-slide median of crop corners) or check the alignment before flat-fielding its LED crops.
4. Non-ideal fluorophore/filter design means underdetermined unmixing (Q4).
5. **Slide-to-slide and batch-to-batch intensity variation** in LED channels of unknown origin (camera or LED array). LED119 is the example.
6. **Focus drift**: "focus could drift by tens of µm over the course of an imaging session". It was handled by per-XY-position single-shot DL autofocus [P §S1.1 "Microscope and data collection"; ref 41]. Histology used a stack-based focus at a subset of positions with interpolation, so per-cell histology focus quality varies [P §S1.1]. No per-cell focus metric is provided [inference].
7. **Regions with debris and slide edges** with optical artifacts were manually excluded before cropping [P §S1.2; Fig S2b]. So missing regions exist by design, but there is no mask or flag in the release.
8. **QC CNN keep/exclude** [P §S1.2; Fig S2f-h, inspected]:
   - Inputs are qDPC + 0.5-0.6 NA darkfield + brightfield into DenseNet201, trained on about 1000 labels; threshold 0.7. Asymptotic test accuracy is about 80%.
   - Test confusion matrix (Fig S2h): **true-keep 185 predicted-keep / 136 predicted-exclude; true-exclude 2 predicted-keep / 303 predicted-exclude**. That is keep-precision ≈ 185/187 = 99%, keep-recall ≈ 58%, accuracy 488/626 = 78% (my arithmetic).
   - So the released set is conservative: few non-cells or doublets, but many good cells discarded.
9. **Selection bias against apoptotic lymphocytes**: "many lymphocytes ... had begun to undergo apoptosis ... characteristic dumbbell-like shape ... cells that looked like this were more likely to be excluded from the final dataset" [P §S1.2]. Lymphocyte morphology and class frequencies are therefore biased.
10. **Non-attached (moving) cells** were removed by darkfield cross-correlation between the first and last channel, about 18 s apart [P §S1.2]. Any residual motion would mis-register channels within a cell's 23-image stack [inference].
11. Only cells **inside the fluorescence field stop** were kept [P §S1.2]. So all cells have valid fluorescence, but positions cluster in the illuminated area.
12. **Unmixing floor and zero-inflation** (Q4). Values at the column minimum are censored. CD123/HLA-DR/CD14 is mostly floor.
13. **Gate labels** cover only batch 0 'all' replicate 0. Authors are unsure that all 10 classes are biological. RBC and Unknown classes exist.
14. **Histology pairing quality is unquantified** (Q1). There is no match-distance column.
15. Documentation and code inconsistencies: Giemsa vs Wright's; the `get_background` "per slide" claim; the `annular_sources` key bug; the 10-bit vs 12-bit histology docstring; the stale-variable bug in the shading LOWESS (Q3b).
15b. **The fluorescence shading correction is ineffective and slightly harmful** [L, A3]. The released `_shading` is anti-correlated with the empirical bright-cell spatial pattern in 5/6 channels, and `_shading_corrected` (and so all `*_unmixed`) is ~5-8% low at the FOV edge (r > 750 px). This is a small effect relative to marker dynamic range, but it is a position-dependent bias. Mitigation: FOV-radius covariate, or use `total_raw − background_lowess`.
16. `detection_radius` comes from the DoG blob finder with permissive parameters [P §S1.2]. It is a coarse, discretised size proxy (value counts in the appendix), not a segmentation.

**Recommended exclusions/handling for N003 (my synthesis):**
- Exclude or flag fluorescence targets from **unstained b1 rep1**.
- Use per-batch target normalisation, and treat floor values as censored.
- Do not use `CD123/HLA-DR/CD14_full` as a supervised target, or down-weight it.
- Do not interpret `CD3/CD19/CD56_full` absolute level without the autofluor baseline. Compare against unstained cells of the same batch.
- Do not use CD3/CD19/CD56 from batch-1 'all' cells for CD19-specific claims.
- For LED inputs, flat-field with `(raw-30)/(bg50-30)` plus a per-slide scale. Consider dropping LED119 or normalising it per batch.
- Hold out a batch for evaluation.
- Measure the histology pairing quality before using histology as a supervised target.

## Q9. Benchmark tasks / splits proposed by the authors

- **No official train/val/test split and no leaderboard** are given in [P], [R] or [D]. `get_indices(shuffle=True, seed=...)` is the only split helper [R `bsccm.py` L264-299].
- Proposed or illustrated uses [P §1, §2.1, §3, §3.1.1]:
  - **Cell-type classification** from label-free images: the 3-class (Lymphocyte/Granulocyte/Monocyte) or 10-class labels via `get_cell_type_classification_data`, batch 0 'all' only. The tutorial's "simple classification problem" [G §4].
  - **Correlating label-free images with surface-protein expression** (protein abundance regression) [P §1, §2.2, Fig 1d].
  - Computational microscopy algorithm development: in-silico illumination design by summing single-LED images, shading correction with the calibration backgrounds, qDPC, Fourier ptychography and diffraction tomography (BSCCM-coherent), self-calibration, autofocus [P §2.1, §3].
  - MNIST-style quick prototyping (BSCCMNIST) [P §3].
- Downstream precedent [I]: the same group predicted **8 protein-marker levels (CD3, CD19, CD56, CD123, HLA-DR, CD14, CD16, CD45)** from LED-array images (Brightfield vs DPC vs single LED), with a composite loss in which "missing values [are] masked out". It used 15 train-test splits for information estimates. This is consistent with using the per-condition single-antibody targets with masking. The exact columns and normalisation were not visible in the parts read [unverified].
- **Recommendation for N003:** define our own splits.
  - (i) A random within-batch split for development.
  - (ii) **Batch-held-out** (train b0 → test b1 and vice versa) as the primary generalisation test.
  - (iii) Ideally slide-level held-out within batch, to separate slide/imaging-date effects. Few conditions have replicate slides.
  - For 3-class classification, only batch 0 'all' replicate 0 is labelled, so use **group-aware CV by FOV (`fov_center_x/y/z`)** to avoid neighbouring-cell leakage [recommendation].

---

## Appendix: first-hand data checks [L]

All checks are read-only on `/hnvme/workspace/v181bb10-bsccm/`. The scripts were run inside `env/bsccm.sif` and live in this session's scratchpad (ephemeral, not committed).

### A1. Selection columns (CSV order), counts, slides
- The surface-marker table has 74 columns.
- It includes an undocumented orphan column **`Fluor_426_shading`**, an intermediate left over in the release; see A3.
- `selection_example_<M>_positive_cells_batch_<b>` (CSV order CD3, HLA-DR, CD16, CD14, CD56, CD19, autofluor, CD123, CD45 for b0; CD3, HLA-DR, CD16, CD14, CD56, CD19, CD123, CD45, autofluor for b1). Each lives only on its own single-antibody slide (rep 0), or on the unstained slide for autofluor. Sizes:

| batch | CD3 | HLA-DR | CD16 | CD14 | CD56 | CD19 | autofluor | CD123 | CD45 |
|---|---|---|---|---|---|---|---|---|---|
| b0 | 872 | 4900 | 4339 | 3270 | 191 | 1476 | 550 | 394 | 5059 |
| b1 | 4251 | 1368 | 15868 | 2920 | 1142 | 1002 | 538 | 524 | 4325 |

- `selection_lymphocytes`: 3025 cells, all in (b0, all, rep0). **2852 of them overlap gate 0** (`selection_gated_CD3_hi_CD_16_lo`, 3037 cells).
- `selection_gated_*` sizes (CSV order = label index 0-9): 3037, 3704, 400, 803, 194, 419, 97, 104, 1225, 180. Total 10,163, all in (b0, all, rep0).
- **10,150 of the 10,210 batch-0 'all' cells (99.4%) carry at least one gate. 13 cells are in more than one gate** (10,163 − 10,150), so `get_cell_type_classification_data` returns those 13 cells twice, possibly with conflicting labels.
- In practice the gating labels essentially all batch-0 'all' cells.
- Implied 3-class counts: Lymphocyte 3037, Granulocyte 3704 + 400 + 803 = 4907, Monocyte 104 + 1225 + 180 = 1509. Dropped: RBC 194, Unknown 516.

### A2. Floor fractions and medians of full-model columns per condition and batch
"floor" = fraction of cells equal to the batch-wide column minimum, i.e. the clipped value. Units are arbitrary.

| condition, batch (n) | CD123/HLA-DR/CD14 floor, med | CD3/CD19/CD56 floor, med | CD45 floor, med | CD16 floor, med |
|---|---|---|---|---|
| all b0 (10,210) | 0.59, 0.11 | 0.00, 87.4 | 0.10, 51.2 | 0.31, 177 |
| all b1 (27,074) | 0.55, 0.09 | 0.09, 70.6 | 0.18, 24.9 | 0.08, 90.6 |
| unstained b0 (16,338) | 0.67, 0.11 | 0.00, 74.5 | 0.15, 8.9 | 0.01, 34.9 |
| unstained b1 (46,729) | 0.50, 0.17 | 0.00, 63.0 | 0.25, 5.8 | 0.01, 37.4 |
| CD45-only b0 (10,951) | 0.67, 0.11 | 0.00, 62.5 | 0.06, 45.9 | 0.07, 34.0 |
| CD45-only b1 (22,033) | 0.16, **10.8** | 0.00, 63.4 | 0.07, 26.6 | 0.07, 19.2 |
| CD16-only b0 (9,397) | 0.73, 0.11 | 0.00, 68.7 | 0.26, 9.2 | 0.04, 54.5 |
| CD16-only b1 (27,857) | 0.42, **10.1** | 0.02, 43.5 | 0.29, 10.9 | 0.10, 87.3 |
| CD3-only b0 (6,538) | 0.74, 0.11 | 0.00, 68.9 | 0.24, 8.1 | 0.02, 36.2 |
| CD3-only b1 (31,332) | 0.55, 0.09 | 0.00, 66.5 | 0.22, 6.5 | 0.05, 36.4 |

What this shows:
- **Unstained cells (no antibody at all) have a median of ~63-75 a.u. on CD3/CD19/CD56, ~35-37 on CD16 and ~6-9 on CD45 in the full model.** This is the autofluorescence and crosstalk baseline the 4-spectrum model leaves in, because it has no autofluor spectrum.
- The 'all' b0 CD3/CD19/CD56 median (87) is barely above unstained (75). The lymphocyte signal is a minority population on top of this baseline.
- Crosstalk depends on batch. CD45-only and CD16-only cells read ~10 a.u. on CD123/HLA-DR/CD14 in b1 but sit at the floor in b0, because the spectra are fitted per batch.
- Recommendation: if the full-model columns are used as targets, calibrate them against the same batch's unstained distribution (e.g. subtract its median, or express as a percentile of unstained).

Single-antibody columns (own condition only), median (p99), from [L]:

| column | b0 | b1 |
|---|---|---|
| CD45 | 75.3 (624) | 37.8 (111) |
| CD123 | 40.8 (61) | 37.6 (48) |
| CD19 | 47.4 (204) | 36.9 (169) |
| CD56 | 20.9 (75) | 9.8 (66) |
| CD14 | 42.7 (105) | 53.4 (74) |
| CD16 | 53.1 (715) | 137.2 (414) |
| HLA-DR | 43.2 (161) | 42.7 (56) |
| CD3 | 16.8 (228) | 1.5 (146) |
| autofluor | 66.2 (170) | 67.8 (214) |

- Floor fraction is ≤ 1.4% for most of them, but **CD56 9%/12% and CD3 9%/43%** (b0/b1).
- Several markers (CD123, CD14, HLA-DR) have narrow dynamic ranges (p99/median ≈ 1.3-3.7), which fits the authors' remark that CD123/HLA-DR/CD14 is noisy.

### A3. Shading check (tests the stale-variable bug in Q3b)
1. **The formula holds exactly.** `_shading_corrected == (_total_raw − _background_lowess) / (_shading / mean(_shading))` for all 6 channels, to a maximum absolute error of 7e-13. So the released columns follow the notebook formula.
2. **Stored shading maps are weak.** Binned-mean coefficient of variation across the FOV is 0.025-0.10. Cell positions span y 180-1916, x 52-1967 px. The maps rise toward the FOV edge: mean shading at r > 750 px vs r < 250 px is +21% (426-446), +8% (500-550), +5% (550-570), +7% (585-625), +5% (627-673), +3% (690-).
3. **Empirical bright-cell vignetting maps vs stored shading.** For each stained slide (excluding unstained and the b1-rep1 slide), I took the per-slide 6x6-bin 90th percentile of `total_raw − background_lowess`, normalised it by the slide median, and averaged across slides (map CV 0.11-0.18). Its correlation with the stored `_shading` map:

| channel | corr(empirical, own stored shading) | corr(empirical, stored 690 shading) |
|---|---|---|
| 426-446 | **−0.66** | 0.08 |
| 500-550 | **−0.74** | −0.05 |
| 550-570 | **−0.68** | −0.07 |
| 585-625 | **−0.81** | 0.00 |
| 627-673 | −0.39 | 0.01 |
| 690- | 0.10 | 0.10 |

   After correction, the same bright-cell map is *less* flat in every channel: CV 0.114→0.155, 0.154→0.170, 0.175→0.183, 0.162→0.174, 0.143→0.153, 0.161→0.163.
4. **Radial test on homogeneous marker-positive populations.** I used the `selection_example_*` gates and each one's dominant channel (426-446 or 500-550), and took the median at r < 250 / 250-500 / 500-750 / > 750 px, normalised to the centre bin.
   - The background-subtracted values are roughly flat (typically within ±5-15%, with no consistent sign).
   - After the shading correction the outer ring drops. Examples (edge value before → after):
     - CD16 b1 (n = 15,868): 1.003 → **0.924**
     - CD16 b0: 1.037 → 0.950
     - CD14 b0: 1.031 → 0.957
     - CD45 b1: 1.013 → 0.942
     - HLA-DR b0: 1.011 → 0.957
     - CD3 b1: 1.037 → 0.997
   - The stored shading is +2-9% at r > 750 px in every case.
   - These gates were drawn on the corrected values, which biases toward a flat corrected profile. So the edge over-correction shows up despite that bias.

**Verdict:**
- The released `_shading` does **not** match the empirical spatial brightness pattern; for 4 of 5 non-690 channels it is anti-correlated.
- The resulting `_shading_corrected` values are slightly *more* position-dependent than the uncorrected background-subtracted values: about 5-8% under-estimation at the FOV edge, r > 750 px, about 27% of cells.
- This is consistent with a broken shading fit (e.g. the stale `channel` variable bug). The exact cause is **[unverified]**, because the 690 channel, which the bug would not affect, also matches poorly.
- **Practical impact is small relative to marker dynamic range** (10-100x, log-scale targets). Mitigations for precise work:
  - use `total_raw − background_lowess` (no shading division);
  - or refit the shading ourselves;
  - or add FOV radius as a nuisance covariate.
- The unmixed `*_model_unmixed` columns inherit this, because they are computed from `_shading_corrected`.
- `Fluor_426_shading` is an undocumented orphan column, fully non-null and correlated 0.84 with `Fluor_426-446_shading`. It is likely an earlier run's leftover. Ignore it.

### A4. Per-slide counts, histology, imaging dates
| batch | antibodies | rep | n cells | histology-matched | imaging_date |
|---|---|---|---|---|---|
| 0 | CD3 | 0 | 6,538 | 0 | 8/4/18 |
| 0 | CD19 | 0 | 12,768 | 0 | 8/5/18 |
| 0 | CD56 | 0 | 14,212 | 0 | 8/7/18 |
| 0 | CD123 | 0 | 18,224 | 0 | 8/8/18 |
| 0 | HLA-DR | 0 | 20,881 | 0 | 8/10/18 |
| 0 | CD45 | 0 | 10,951 | 0 | 8/13/18 |
| 0 | CD14 | 0 | 18,442 | 0 | 8/15/18 |
| 0 | CD16 | 0 | 9,397 | 0 | 8/16/18 |
| 0 | unstained | 0 | 16,338 | **10,015 (61%)** | 8/19/18 |
| 0 | all | 0 | 10,210 | 0 | 8/21/18 |
| 1 | unstained | 0 | 32,674 | 0 | 8/27/18 |
| 1 | all | 0 | 13,675 | 0 | 8/28/18 |
| 1 | CD16 | 0 | 27,857 | 0 | 8/30/18 |
| 1 | CD14 | 0 | 25,223 | 0 | 9/10/18 |
| 1 | CD45 | 0 | 22,033 | 0 | 9/12/18 |
| 1 | HLA-DR | 0 | 32,965 | 0 | 9/13/18 |
| 1 | CD19 | 0 | 7,736 | 0 | 9/14/18 |
| 1 | CD3 | 0 | 31,332 | 0 | 9/15/18 |
| 1 | CD56 | 0 | 23,141 | 0 | 9/17/18 |
| 1 | all | 1 | 13,399 | 0 | 9/18/18 |
| 1 | unstained | 1 | 14,055 | **5,697 (41%)** | 9/20/18 |
| 1 | CD123 | 0 | 30,890 | 0 | 9/21/18 |

Totals: b0 137,961; b1 274,980; all 412,941.

Facts that follow (all [L]):
- **Exactly two slides were histology-stained: the unstained slides (b0, rep0) and (b1, rep1).** 15,712 = 10,015 + 5,697.
- Within those slides, only 61% and 41% of the LED cells got a histology match. **That is a direct lower bound on cell loss and non-matching from the disassembly, staining and pairing.**
- **The batch-1 histology slide is the same slide the authors flag as imaged without the camera ROI** [P §S1.5 bullet 3] and with a suspect field stop (`shading_correction.ipynb` cell 14).
- Each slide is one condition on one date, so **slide = condition = imaging date (confounded)**.
- Imaging spans 17 days for batch 0 (8/4 to 8/21/18) and 25 days for batch 1 (8/27 to 9/21/18), longer than the "next 2 weeks" in the paper. The isolation dates are not recorded [unverified]. If isolation was just before the first imaging date, cells sat fixed for up to ~3.5 weeks.
- `detection_radius` takes only 4 values: 20.48 (352,809 cells), 12.8 (58,536), 32.768 (1,574), 8.0 (22). These are DoG scale levels, so it is a ~3-level size proxy only.

### A5. DPC check
Method: I wrote my own minimal qDPC, not running repo code. It uses the weak-object transfer function for 4 half-annulus sources (NA 0.40-0.50, objective NA 0.5, λ = 0.515 µm, 0.166 µm pixels), Tikhonov regularisation (0.1 / 5e-3), and offset-30 + per-crop mean flat-field. It was run on 40 random cells from `pack/led_array.npy` channels 18-21 and correlated with `pack/dpc.npy`.
- **corr(my phase, stored `dpc`) = 0.984 median (min 0.970, max 0.990).**
- corr(my absorption, stored) = −0.45.
- corr(naive (L−R)/(L+R) or (T−B)/(T+B), stored) ≈ 0.06 / 0.02.
- corr(Brightfield, stored) = −0.69.
- **Conclusion: `dpc` is the quantitative-phase output of a WOTF/Tikhonov qDPC inversion (Tian & Waller 2015) of the 4 DPC channels.** It is not a raw DPC difference and not the absorption term.
- The small residual difference is consistent with the authors reconstructing on the full 2056² FOV with the measured flat-field and their own regularisation.
- Value statistics on 1000 random cells per batch:
  - b0: mean 0.066, sd 0.75, p1/p99 −0.76/2.66, range [−1.37, 3.95], 4-px border mean −0.09, central 20x20 mean 2.04.
  - b1: mean 0.055, sd 0.73, p1/p99 −0.76/2.64, range [−1.27, 4.07], border −0.07, central 2.04.
- This is consistent with **phase in radians at 515 nm** (cell ≈ 2 rad, background ≈ 0) [inference; units are not documented].
- The two batches have very similar distributions, so `dpc` looks batch-robust, unlike `LED119`.

### Code quirks found (all [R] at 64de2e0)
- `bsccm.py` L344-362: `get_background` docstring says per slide, but the code reads one global array per channel. Crops near the FOV border are truncated (no padding).
- `bsccm.py` L364-405: labels depend on CSV column order. There is no overlap check.
- `bsccm.py` L214-215 vs L258: the docstring says 10-bit, the code divides by 2^12.
- `phase/functional_dpc.py` L112 vs L40: `'NAinner'` vs `'NA inner'` key mismatch, so the tutorial's DPC sources ignore the inner NA.
- `shading_correction.ipynb` cell 20 (JSON L1264-1265): positions are taken from a stale `channel` variable (= `'Fluor_690-'`) for every channel's shading LOWESS. See A3 for whether the release shows it.
- `do_demixing.ipynb`: the background B is shared per (antibodies x batch), while the paper says per slide. The paper's "λ=5e-2" means β, and the code uses an L2 (not L1) penalty on B (`demixing_util.py` L206).
- `compute_fluorescence.ipynb`: uses mean, not sum (the paper says "summing").
