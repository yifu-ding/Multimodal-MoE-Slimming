# Observation B: single-expert second-order validation

This directory contains the publication-facing figure and the compact real
data needed to reproduce it. Observation A and the previous pairwise 3D
landscape are **LEGACY / NOT USED** by this figure.

## What the figure shows

- Panels (a)-(b): across 3,968 layer-expert directions, the autograd HVP score
  `H_ee/2` agrees with routed expert-output energy and measured single-expert
  ablation error.
- Panel (c): the numerical relative-error distribution for those identities.
- Panels (d)-(f): real beta forwards for low-, medium-, and high-sensitivity
  experts in layer 0. The first-order prediction is zero at the identity
  reconstruction point, while the diagonal-Hessian parabola follows every
  measured point, including single-expert removal at `beta_e=0`.

No pairwise expert removal is used.

The frozen data covers 31 completed MoE layers, all 128 routed experts in each
layer, and the first 32 samples of the calibration manifest. Running every
model layer is not required here: this is a pointwise algebraic validation,
not an estimate of a population mean.

## What one data point means

Each point in panels (a) and (b) is one `(layer, expert)` pair, aggregated over
the same 32 samples and normalized by the total number of scored tokens. Its
x-coordinate is the autograd HVP result

```text
hvp_hessian_half = (1 / 2) * d^2 L / d beta_e^2 at beta_e = 1.
```

Its y-coordinate is either `expert_output_energy` in panel (a), or the output
MSE caused by setting only that expert's scale to zero,
`single_expert_ablation`, in panel (b). Routing is held fixed. A point on the
red `y=x` line therefore says that this expert's diagonal-Hessian score exactly
predicts its single-expert removal cost.

Each circle in panels (d)-(f) is one real block forward for one selected layer-0
expert at one `beta` value. The three experts are the P10, P50, and P90 entries
of the layer-0 `H_ee/2` ranking. The red diamond is the circle at `beta=0`, i.e.
single-expert removal. These are measured points, not samples from the plotted
parabola.

## Data dictionary

`data/method_validation_B_scores.csv` has one row per active `(layer, expert)`:

- `layer`, `expert`: zero-based MoE layer and expert IDs.
- `hvp_hessian_half`: diagonal second-order score `H_ee/2`.
- `expert_output_energy`: squared routed expert contribution, using the same
  hidden-dimension MSE and score-token normalization.
- `single_expert_ablation`: exact output MSE constructed from the actual routed
  contribution after removing only this expert; no quadratic formula is used.
- `identity_gradient`: `dL/d beta_e` at `beta_e=1`; it is zero here because
  identity reconstruction is the MSE minimum.
- `active_batch_count`: number of the two validation batches in which the
  expert received at least one scored token.

`data/method_validation_B_beta_curves.csv` has one row per real beta forward:

- `sensitivity`, `quantile`: low/P10, medium/P50, or high/P90 selection.
- `layer`, `expert`, `beta`: the scaled expert direction and evaluated scale.
- `measured_delta_mse`: measured MSE increase relative to `beta=1`.
- `first_order_prediction`: `g_e * (beta-1)`.
- `second_order_prediction`: `g_e * (beta-1) + (H_ee/2) * (beta-1)^2`.
- `identity_gradient`, `hvp_hessian_half`: the coefficients used by those two
  predictions.

`data/method_validation_B_metadata.json` records the model, exact manifest
hash, layer list, sample count, normalization, FP32 validation precision, and
the fixed-router check.

## Reproduce the figure

From the repository root:

```bash
python draw/hessian-3d-landscape/plot_method_validation_b.py
```

This reads only:

```text
draw/hessian-3d-landscape/data/method_validation_B_scores.csv
draw/hessian-3d-landscape/data/method_validation_B_beta_curves.csv
draw/hessian-3d-landscape/data/method_validation_B_metadata.json
```

and writes `method_validation_B.pdf`, `method_validation_B.png`, and a metric
summary `method_validation_B.json` in this directory.

## Recollect or export

The four-GPU collector uses FP32 copied blocks for strict numerical agreement:

```bash
LIVE_LOGS=0 FORCE=1 VALIDATION_SAMPLES=32 \
bash scripts/run_method_validation_b_4gpu.sh
```

It atomically checkpoints completed layers. An interrupted run can be resumed
with `RESUME=1 RUN_SMOKE_TEST=0`; resume mode validates collection metadata
before skipping any layer.

Convert a merged collector payload into the compact public data:

```bash
python scripts/export_method_validation_b_csv.py \
  --input <output-dir>/method_validation_b.pt \
  --output-dir draw/hessian-3d-landscape/data
```

The acceptance check is:

```bash
python scripts/check_method_validation_b.py \
  --input <output-dir>/method_validation_b.pt
```
