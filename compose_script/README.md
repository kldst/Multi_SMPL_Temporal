# MAMMA 518 compose cache

The converter uses the production principal-point crop/resize, transforms the
instance mask identically, and stores the adjusted camera intrinsics. It never
modifies or copies the large source `*.data.pyd` files; compact SMPL/camera/person
metadata is written to one `manifest.pkl` per sequence.

Full resumable conversion:

```bash
/train-data-3-hdd/yian/conda/envs/mamma/bin/python \
  compose_script/compose_mamma_518.py
```

Deterministic smoke subset:

```bash
/train-data-3-hdd/yian/conda/envs/mamma/bin/python \
  compose_script/compose_mamma_518.py \
  --dataset moyo_4-6_C_200_00 \
  --sequence be_sf-zEiRjQqSO_seq_000000 \
  --max-frames 4
```

Output layout:

```text
mamma_compose/<dataset>/<sequence>/
  manifest.pkl
  IOI_05/
    0003.png
    0003.mask.png
    0003.camera.npz
```

`*.camera.npz` contains the post-crop/resize `intrinsics`, unchanged
`extrinsics`, source `original_size`, and the legacy half-pixel `track_offset`
needed to reproduce the original 2D GT/loss exactly. Existing complete bundles
are reused; later running without `--max-frames` extends the same sequence
manifest instead of discarding the smoke subset.

Do not point a full training run at a partial cache. Enable
`use_mamma_compose_cache` in the YAML only after the unrestricted conversion has
finished and `compose_complete.json` exists. The production YAML requires this
completion marker; the debug validator explicitly allows its four-frame subset.
