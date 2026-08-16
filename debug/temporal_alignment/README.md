# Temporal SMPL one-batch alignment check

This folder validates the complete data/layout boundary used by
`mamma_harmony4d_mask_dpt.yaml`. It uses a real temporal clip rather than a
synthetic tensor.

Run the inexpensive data, reprojection, and GT-loss check:

```bash
/train-data-3-hdd/yian/conda/envs/mamma/bin/python \
  debug/temporal_alignment/check_temporal_batch.py \
  --skip-model-forward
```

Run the same check plus one configured temporal VGGT forward and prediction
loss (random weights if no checkpoint is supplied):

```bash
CUDA_VISIBLE_DEVICES=1 \
/train-data-3-hdd/yian/conda/envs/mamma/bin/python \
  debug/temporal_alignment/check_temporal_batch.py
```

To test a trained checkpoint, append:

```bash
--checkpoint /absolute/path/to/checkpoint.pt
```

Outputs are saved to `outputs/one_batch/`:

- `result.json`: shapes, temporal contract checks, reprojection metrics, mask
  statistics, GT-as-pred loss, and model prediction loss;
- `config_resolved.yaml`: the exact one-batch config after local path overrides;
- `overlays/`: per-frame/per-view images and contact sheets. Green circles are
  stored 2-D joints, red crosses are the corresponding 3-D joints reprojected
  using the batch camera, and colored regions are person-mask GT.

## Layout being checked

The loader returns one clip in time-major order:

```text
images / camera / view GT: [B, T*V, ...] = [1, 24, ...]
SMPL person parameters:     [B, T, P, ...] = [1, 3, 6, ...]
frame_ids:                  [B, T]
view_ids:                   [B, V]
```

The temporal VGGT path reshapes images to `[B*T, V, ...]` only around the VGGT
aggregator. Therefore global attention sees the eight simultaneous views of one
timestep and cannot mix unrelated cameras across time. Its patch context is
restored to `[B, T, V*Npatch, C]` for the temporal SMPL decoder.

The decoder creates persistent person slots `[B, P, T, D]`. Each slot attends
along its own trajectory using temporal RoPE, and cross-attends to shared
spatio-temporal image context using relative temporal bias. It predicts
`[B, T, P, ...]`, then exports `[B*T, P, ...]` so the existing camera/SMPL/mask
loss interface remains unchanged.

Trainer applies camera gauge normalization before forward, but does not flatten
the clip until after the temporal head has run. The loss then performs
frame-level Hungarian matching, uses the matched GT identity order for temporal
pose/beta/translation regularization, and evaluates all existing framewise
supervision on `[B*T, V, ...]`.

The full-resolution DPT mask trunk uses `frames_chunk_size: 1`. Temporal folding
makes its effective batch `B*T`; processing one view at a time avoids a large
interpolation peak on a 24GB GPU. Chunking does not add or remove parameters and
does not change the tensor contract or checkpoint keys.

GT-as-pred validates the supervision wiring. Its camera, SMPL parameters,
mesh-translation, 3-D joints, and vertices should be numerically near zero.
Temporal losses are smoothness regularizers on the real motion itself, so pose
or translation terms may correctly remain nonzero even for GT-as-pred.
