import logging
import os
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import read_depth, read_image_cv2
from training.data.landmark_mask_gt import (
    NUM_SMPLX_VERTS,
    downsample_vertices,
    downsample_visibility,
    rasterize_person_patch_mask,
)


class SysSMPLMultiDataset(BaseDataset):
    """
    Dataset for multi-person raw Mamma_mv_split scenes.

    One multi-person, multi-view sample = the same time step (frame) seen from
    ``img_per_seq`` views. Ground truth is read from the per-view ``.data.pyd``
    (SMPL-X world params, per-camera intrinsics/extrinsics, projected
    ``vertices2d`` / ``vertex_visibility``) plus the per-view instance mask
    ``*.mask.jpg`` (pixel value == person_idx + 1). Layout:

        <root>/.../png/<seq>/<IOI_view>/<frame>.jpg  (+ .data.pyd, .mask.jpg)

    Dense-landmark (``smpl_landmarks2d`` / ``_visibility``) and per-person mask
    (``person_mask``) GT are emitted when ``emit_landmarks`` / ``emit_person_mask``
    are set (derived on the fly via the ``verts_512`` matrix).
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SysSMPL_DIR: str = None,
        SysSMPL_ANNOTATION_DIR: str = None,
        min_num_images: int = 20,
        max_num_people: Optional[int] = None,
        len_train: Optional[int] = None,
        emit_landmarks: bool = False,
        emit_person_mask: bool = False,
        # Resolution of the emitted person_mask GT: mask grid = processed image
        # size // person_mask_stride.  None keeps the legacy patch-grid behaviour
        # (stride = patch_size -> 37x37 for 518/14).  Set 2 to supervise a
        # pixel-level mask head (259x259), matching model.person_mask_down_ratio.
        person_mask_stride: Optional[int] = None,
        emit_contact: bool = False,
        contact_threshold: float = 0.01,
        # When sdf_vertices is absent for a frame, how to label person-person contact:
        #   True  (MAMMA): treat as 0 (no-contact) -> supervised as negatives.
        #   False        : mark -1 -> ignored by the loss (no supervision).
        # Empirically (debug/check_contact_sdf_availability.py) missing-sdf frames are
        # genuinely non-contact (people >0.5m apart), so True is the justified default.
        contact_missing_as_negative: bool = True,
        landmark_matrix_path: Optional[str] = None,
        landmark_visibility_threshold: float = 0.5,
        max_sequences: Optional[int] = None,
        max_frames_per_sequence: Optional[int] = None,
        compose_cache_root: Optional[str] = None,
        prefer_compose_cache: bool = False,
        require_complete_compose_cache: bool = True,
    ):
        super().__init__(common_conf=common_conf)
        # Optional caps to bound the (raw Mamma) build, which cold-reads one pyd
        # per (frame,view). Default None = load everything. Set small (e.g. 1-2
        # sequences) for fast debug / quick-iteration startup on a cold disk.
        self.max_sequences = max_sequences
        self.max_frames_per_sequence = max_frames_per_sequence
        self.compose_cache_root = (
            Path(compose_cache_root).expanduser()
            if compose_cache_root is not None
            else None
        )
        self.prefer_compose_cache = bool(prefer_compose_cache)
        self.require_complete_compose_cache = bool(require_complete_compose_cache)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.fixed_view_sampling = getattr(common_conf, "fixed_view_sampling", False)
        self.use_temporal_training = bool(
            getattr(common_conf, "use_temporal_training", False)
        )
        self.temporal_clip_length = int(
            getattr(common_conf, "temporal_clip_length", 3)
        )
        self.temporal_clip_stride = int(
            getattr(common_conf, "temporal_clip_stride", 1)
        )
        if self.temporal_clip_length < 2:
            raise ValueError("temporal_clip_length must be >= 2")
        if self.temporal_clip_stride < 1:
            raise ValueError("temporal_clip_stride must be >= 1")

        if SysSMPL_DIR is None or SysSMPL_ANNOTATION_DIR is None:
            raise ValueError("SysSMPL_DIR and SysSMPL_ANNOTATION_DIR must be specified.")

        self.image_root = Path(SysSMPL_DIR).expanduser()
        # Both roots point at a raw Mamma_mv_split split root; the loader scans
        # for png/<seq>/<IOI_view>/<frame>.data.pyd under them.
        self.data_root = Path(SysSMPL_ANNOTATION_DIR).expanduser()

        if not self.image_root.is_dir():
            raise ValueError(f"Image root not found: {self.image_root}")
        if not self.data_root.is_dir():
            raise ValueError(f"Merged out_data root not found: {self.data_root}")

        self.min_num_images = min_num_images
        self.max_num_people = max_num_people

        # Dense-landmark / per-person-mask GT (raw Mamma_mv_split only). Both are
        # derived on the fly from data the pyd already ships (vertices2d /
        # vertex_visibility / *.mask.jpg) via the ``verts_512`` matrix.
        self.emit_landmarks = bool(emit_landmarks)
        self.emit_person_mask = bool(emit_person_mask)
        self.person_mask_stride = (
            int(person_mask_stride) if person_mask_stride is not None else None
        )
        # MAMMA-style per-landmark contact GT (person-person via sdf_vertices + floor
        # via floor_contact_mask). Needs the verts_512 matrix, so it also loads it.
        self.emit_contact = bool(emit_contact)
        self.contact_threshold = float(contact_threshold)
        self.contact_missing_as_negative = bool(contact_missing_as_negative)
        self.landmark_visibility_threshold = float(landmark_visibility_threshold)
        self.patch_grid = int(self.img_size // self.patch_size)
        self._verts512 = None
        if self.emit_landmarks or self.emit_contact:
            from training.data.landmark_mask_gt import load_verts512_matrix
            self._verts512 = load_verts512_matrix(landmark_matrix_path)

        if self.prefer_compose_cache:
            if self.compose_cache_root is None or not self.compose_cache_root.is_dir():
                raise ValueError(
                    "prefer_compose_cache=true requires an existing compose_cache_root; "
                    f"got {self.compose_cache_root}"
                )
            if self.emit_landmarks or self.emit_contact:
                raise ValueError(
                    "The compact 518 compose cache does not retain dense vertices/SDF; "
                    "emit_landmarks and emit_contact must both be false."
                )
            if self.emit_dense_geometry:
                raise ValueError(
                    "The 518 compose cache fast path requires emit_dense_geometry=false."
                )
            if (
                self.require_complete_compose_cache
                and not (self.compose_cache_root / "compose_complete.json").is_file()
            ):
                raise ValueError(
                    "Compose cache is partial or unfinished: expected "
                    f"{self.compose_cache_root / 'compose_complete.json'}. Run the "
                    "converter without --dataset/--sequence/--max-* before training."
                )

        self.data_store = self._build_sequences()
        people_source = (
            self.frame_data_store.values()
            if self.use_temporal_training
            else self.data_store.values()
        )
        inferred_max_people = max(
            (view_annos[0]["num_people"] for view_annos in people_source),
            default=0,
        )
        if self.max_num_people is None:
            self.max_num_people = inferred_max_people
        elif inferred_max_people > self.max_num_people:
            # get_data already truncates with min(person_count, max_num_people), so a
            # crowded scene should cost you the extra people, not the whole training
            # run.  Raising here used to kill startup outright.
            logging.warning(
                "SysSMPLMulti: observed %d people but max_num_people=%d; the extra "
                "people will be TRUNCATED. Raise max_num_people (and model."
                "smpl_num_people) to keep them.",
                inferred_max_people, self.max_num_people,
            )
        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.use_temporal_training:
            self.total_frame_num = sum(
                len(clip["frame_keys"]) * len(clip["common_view_names"])
                for clip in self.data_store.values()
            )
        else:
            self.total_frame_num = sum(len(seq) for seq in self.data_store.values())

        if split == "train":
            self.len_train = self.sequence_list_len if len_train is None else min(len_train, self.sequence_list_len)
        elif split == "test":
            self.len_train = self.sequence_list_len
        else:
            raise ValueError(f"Invalid split: {split}")

        status = "Training" if self.training else "Testing"
        logging.info("%s: SysSMPLMulti sequences: %d", status, self.sequence_list_len)
        logging.info("%s: SysSMPLMulti total views: %d", status, self.total_frame_num)
        logging.info("SysSMPLMulti max_num_people: %d", self.max_num_people)
        logging.info("SysSMPLMulti data_root: %s", self.data_root)
        logging.info(
            "SysSMPLMulti compose cache: enabled=%s, root=%s",
            self.prefer_compose_cache,
            self.compose_cache_root,
        )
        logging.info(
            "SysSMPLMulti temporal training: enabled=%s, clip_length=%d, stride=%d",
            self.use_temporal_training,
            self.temporal_clip_length,
            self.temporal_clip_stride,
        )

    @staticmethod
    def _load_pickle(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _parse_camera(intrinsics, extrinsics) -> Optional[Dict[str, np.ndarray]]:
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        extrinsics = np.asarray(extrinsics, dtype=np.float32)

        if intrinsics.ndim == 1 and intrinsics.size == 9:
            intrinsics = intrinsics.reshape(3, 3)
        elif intrinsics.shape != (3, 3):
            return None

        if extrinsics.ndim == 1 and extrinsics.size == 12:
            extrinsics = extrinsics.reshape(3, 4)
        elif extrinsics.shape == (4, 4):
            extrinsics = extrinsics[:3, :]
        elif extrinsics.shape != (3, 4):
            return None

        return {"intrinsics": intrinsics, "extrinsics": extrinsics}

    @staticmethod
    def _project_points_opencv_np(points_world: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        E = np.asarray(extrinsics, dtype=np.float64).reshape(3, 4)
        K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        cam = points @ E[:3, :3].T + E[:3, 3]
        z = cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        pix = (cam @ K.T)[:, :2] / z_safe[:, None]
        return pix.astype(np.float32)

    @staticmethod
    def _looks_like_raw_mamma_sequence(path: Path) -> bool:
        if not path.is_dir():
            return False
        for view_dir in path.iterdir():
            if view_dir.is_dir() and any(view_dir.glob("*.data.pyd")):
                return True
        return False

    @classmethod
    def _find_raw_mamma_sequences(cls, root: Path) -> List[Path]:
        root = Path(root)
        if cls._looks_like_raw_mamma_sequence(root):
            return [root]

        seq_dirs = []
        png_roots = []
        if (root / "png").is_dir():
            png_roots.append(root / "png")
        png_roots.extend(p for p in root.rglob("png") if p.is_dir())
        for png_root in sorted(set(png_roots)):
            for child in sorted(p for p in png_root.iterdir() if p.is_dir()):
                if cls._looks_like_raw_mamma_sequence(child):
                    seq_dirs.append(child)
            if cls._looks_like_raw_mamma_sequence(png_root):
                seq_dirs.append(png_root)
        if seq_dirs:
            return sorted(set(seq_dirs))

        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            if cls._looks_like_raw_mamma_sequence(current):
                seq_dirs.append(current)
                dirnames[:] = []
                continue
            if current.name != "png" and current.parent.name != "png":
                # Keep traversal broad enough for Mamma_mv_split/.../png/<seq>,
                # but avoid descending into image leaves once possible.
                pass
        return sorted(set(seq_dirs))

    @staticmethod
    def _group_raw_mamma_frames(seq_dir: Path) -> Dict[str, List[str]]:
        frame_to_views: Dict[str, List[str]] = {}
        for view_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
            for data_path in view_dir.glob("*.data.pyd"):
                frame = data_path.name[: -len(".data.pyd")]
                if (view_dir / f"{frame}.jpg").is_file() or (view_dir / f"{frame}.png").is_file():
                    frame_to_views.setdefault(frame, []).append(view_dir.name)
        return {frame: sorted(views) for frame, views in frame_to_views.items()}

    @staticmethod
    def _raw_image_path(seq_dir: Path, view_name: str, frame: str) -> Optional[Path]:
        view_dir = seq_dir / view_name
        for suffix in (".jpg", ".png", ".jpeg"):
            image_path = view_dir / f"{frame}{suffix}"
            if image_path.is_file():
                return image_path
        return None

    def _decode_raw_mamma_joints_world_batch(self, people) -> np.ndarray:
        # Import lazily to keep the classic npz loader lightweight and avoid a
        # module-level dependency from data loading to the loss stack.
        import inspect
        import torch

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
        from training.loss import _decode_smpl_batch, _normalize_gender_string

        pose_t = torch.as_tensor(np.stack([
            np.asarray(person["smpl_pose"], dtype=np.float32).reshape(-1)[:72]
            for person in people
        ]))
        beta_t = torch.as_tensor(np.stack([
            np.asarray(person["smpl_beta"], dtype=np.float32).reshape(-1)[:10]
            for person in people
        ]))
        trans_t = torch.as_tensor(np.stack([
            np.asarray(person["smpl_trans"], dtype=np.float32).reshape(3)
            for person in people
        ]))
        genders = [
            _normalize_gender_string(person.get("gender", "neutral"))
            for person in people
        ]
        with torch.no_grad():
            joints, _ = _decode_smpl_batch(
                pose_t,
                beta_t,
                trans_t,
                genders,
                use_mamma=True,
                with_vertices=False,
            )
        return joints[:, :24].detach().cpu().numpy().astype(np.float32)

    def _build_sequences(self) -> Dict[str, List[Dict[str, np.ndarray]]]:
        if self.prefer_compose_cache:
            frame_store, sequence_frames = self._build_composed_mamma_sequences()
            source_description = f"518 compose manifests under {self.compose_cache_root}"
        else:
            frame_store, sequence_frames = self._build_raw_mamma_sequences()
            source_description = f"raw Mamma data under {self.data_root} / {self.image_root}"
        if not frame_store:
            raise ValueError(f"No usable frames found from {source_description}.")
        logging.info("SysSMPLMulti Mamma indexed frames: %d (%s)", len(frame_store), source_description)
        self.frame_data_store = frame_store
        if not self.use_temporal_training:
            return frame_store
        clip_store = self._build_temporal_clip_index(frame_store, sequence_frames)
        if not clip_store:
            raise ValueError(
                "Temporal training is enabled but no consecutive clips with enough "
                "shared views were found."
            )
        logging.info("SysSMPLMulti temporal clips: %d", len(clip_store))
        return clip_store

    def _build_composed_mamma_sequences(self):
        """Build an index from compact 518 manifests without reading raw pyd files."""
        manifest_paths = sorted(self.compose_cache_root.rglob("manifest.pkl"))
        if self.max_sequences is not None:
            manifest_paths = manifest_paths[: int(self.max_sequences)]

        data_store: Dict[str, List[Dict[str, np.ndarray]]] = {}
        sequence_frames = defaultdict(list)
        expected_shape = tuple(int(x) for x in self.get_target_shape(1.0))
        skipped = defaultdict(int)

        for manifest_path in manifest_paths:
            try:
                manifest = self._load_pickle(manifest_path)
            except Exception as exc:
                logging.warning("SysSMPLMulti: unreadable compose manifest %s: %s", manifest_path, exc)
                skipped["bad_manifest"] += 1
                continue
            if (
                not isinstance(manifest, dict)
                or manifest.get("format") != "mamma_compose_518"
                or int(manifest.get("version", -1)) != 1
            ):
                skipped["wrong_format"] += 1
                continue
            manifest_shape = tuple(int(x) for x in manifest.get("target_shape", ()))
            if manifest_shape != expected_shape:
                raise ValueError(
                    f"Compose cache {manifest_path} has target_shape={manifest_shape}, "
                    f"but this dataloader requires {expected_shape}."
                )

            sequence_name = str(
                manifest.get("sequence_rel")
                or manifest_path.parent.relative_to(self.compose_cache_root)
            )
            frame_items = sorted(
                manifest.get("frames", {}).items(),
                key=lambda item: self._numeric_frame_id(item[0]),
            )
            if self.max_frames_per_sequence is not None:
                frame_items = frame_items[: int(self.max_frames_per_sequence)]

            for frame, manifest_annos in frame_items:
                view_annos = []
                for cached in manifest_annos:
                    image_path = manifest_path.parent / cached["image_path"]
                    mask_rel = cached.get("mask_path")
                    mask_path = manifest_path.parent / mask_rel if mask_rel else None
                    camera_path = manifest_path.parent / cached["camera_path"]
                    if not image_path.is_file() or not camera_path.is_file():
                        skipped["missing_bundle"] += 1
                        continue
                    if mask_path is not None and not mask_path.is_file():
                        skipped["missing_bundle"] += 1
                        continue
                    try:
                        with np.load(camera_path) as camera:
                            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
                            extrinsics = np.asarray(camera["extrinsics"], dtype=np.float32)
                            original_size = np.asarray(camera["original_size"], dtype=np.int64)
                            track_offset = np.asarray(camera["track_offset"], dtype=np.float32)
                    except Exception:
                        skipped["bad_camera_bundle"] += 1
                        continue
                    if (
                        intrinsics.shape != (3, 3)
                        or extrinsics.shape != (3, 4)
                        or track_offset.shape != (2,)
                    ):
                        skipped["bad_camera_bundle"] += 1
                        continue

                    view_annos.append(
                        {
                            "view_name": str(cached["view_name"]),
                            "image_path": str(image_path),
                            "mask_path": str(mask_path) if mask_path is not None else None,
                            "data_path": cached.get("source_data_path"),
                            "intrinsics": intrinsics,
                            "extrinsics": extrinsics,
                            "original_size": original_size,
                            "track_offset": track_offset,
                            "people": cached["people"],
                            "num_people": int(cached["num_people"]),
                            "raw_mamma": True,
                            "composed_mamma": True,
                            "preprocessed_518": True,
                        }
                    )
                if len(view_annos) < self.min_num_images:
                    skipped["too_few_views"] += 1
                    continue
                if len({anno["num_people"] for anno in view_annos}) != 1:
                    skipped["inconsistent_num_people"] += 1
                    continue
                frame_key = f"compose_mamma_{sequence_name}_frame_{frame}"
                data_store[frame_key] = view_annos
                numeric_frame_id = self._numeric_frame_id(frame)
                sequence_frames[sequence_name].append(
                    (numeric_frame_id, str(frame), frame_key)
                )

        logging.info(
            "SysSMPLMulti compose summary: manifests=%d frames=%d skipped=%s",
            len(manifest_paths),
            len(data_store),
            dict(skipped),
        )
        return data_store, dict(sequence_frames)

    @staticmethod
    def _numeric_frame_id(frame_name: str) -> int:
        """Extract the last integer from a MAMMA frame basename."""
        matches = re.findall(r"\d+", str(frame_name))
        if not matches:
            raise ValueError(f"Frame name has no numeric id: {frame_name!r}")
        return int(matches[-1])

    def _build_temporal_clip_index(self, frame_store, sequence_frames):
        """Build fixed-length clips without crossing a physical frame-number gap."""
        clip_store = {}
        T = self.temporal_clip_length
        stride = self.temporal_clip_stride
        for sequence_name, records in sequence_frames.items():
            ordered = sorted(records, key=lambda item: item[0])
            runs = []
            for record in ordered:
                if not runs or record[0] - runs[-1][-1][0] != 1:
                    runs.append([])
                runs[-1].append(record)

            for run in runs:
                if len(run) < T:
                    continue
                for start in range(0, len(run) - T + 1, stride):
                    window = run[start : start + T]
                    frame_keys = [record[2] for record in window]
                    view_sets = [
                        {anno["view_name"] for anno in frame_store[key]}
                        for key in frame_keys
                    ]
                    common_views = sorted(set.intersection(*view_sets))
                    if len(common_views) < self.min_num_images:
                        continue
                    frame_ids = [record[0] for record in window]
                    if any(b - a != 1 for a, b in zip(frame_ids, frame_ids[1:])):
                        raise RuntimeError(f"Non-consecutive temporal clip: {frame_ids}")
                    clip_key = (
                        f"temporal_{sequence_name}_frames_"
                        f"{frame_ids[0]:06d}_{frame_ids[-1]:06d}"
                    )
                    clip_store[clip_key] = {
                        "sequence_name": sequence_name,
                        "frame_keys": frame_keys,
                        "frame_ids": frame_ids,
                        "common_view_names": common_views,
                    }
        return clip_store

    def _build_raw_mamma_sequences(self):
        roots = [self.data_root]
        if self.image_root != self.data_root:
            roots.append(self.image_root)

        seq_dirs = []
        for root in roots:
            if root.is_dir():
                seq_dirs.extend(self._find_raw_mamma_sequences(root))
        seq_dirs = sorted(set(seq_dirs))
        if self.max_sequences is not None:
            seq_dirs = seq_dirs[: int(self.max_sequences)]
        if not seq_dirs:
            return {}, {}

        data_store: Dict[str, List[Dict[str, np.ndarray]]] = {}
        sequence_frames = defaultdict(list)
        bad_pyds: List[str] = []
        # Every frame that does NOT make it into data_store is counted here, so a
        # silently-empty dataset is visible in the log instead of looking like a
        # successful build (see the per-sequence summary at the end).
        drops: Dict[str, int] = defaultdict(int)
        per_seq_kept: Dict[str, int] = {}
        per_seq_seen: Dict[str, int] = {}
        total_seqs = len(seq_dirs)
        logging.info(
            "SysSMPLMulti: building index over %d sequences (serial, reads 1 pyd per frame/view)...",
            total_seqs,
        )
        for seq_i, seq_dir in enumerate(seq_dirs, 1):
            grouped = self._group_raw_mamma_frames(seq_dir)
            frame_items = sorted(grouped.items())
            if self.max_frames_per_sequence is not None:
                frame_items = frame_items[: int(self.max_frames_per_sequence)]
            kept_here = 0
            per_seq_seen[str(seq_dir)] = len(frame_items)
            for frame, views in frame_items:
                if len(views) < self.min_num_images:
                    drops["too_few_views"] += 1
                    continue

                # Read every view's pyd ONCE up front.  The people set is then the
                # UNION over views, not whatever the alphabetically-first view
                # happened to annotate: the SMPL params are world-frame and are
                # byte-identical across views, so a person occluded in one view is
                # still fully supervised in 3D from the others.  Only that view's
                # 2D targets get switched off (``visible_in_view`` below).
                view_people = []
                for view_name in views:
                    data_path = seq_dir / view_name / f"{frame}.data.pyd"
                    image_path = self._raw_image_path(seq_dir, view_name, frame)
                    if image_path is None or not data_path.is_file():
                        continue
                    try:
                        frame_people = self._load_pickle(data_path)
                    except Exception:  # corrupt/unreadable pyd -> skip this view
                        bad_pyds.append(str(data_path))
                        continue
                    if not isinstance(frame_people, dict) or not frame_people:
                        continue
                    camera = self._parse_camera(
                        next(iter(frame_people.values()))["cam_int"],
                        next(iter(frame_people.values()))["cam_ext"],
                    )
                    if camera is None:
                        drops["bad_camera"] += 1
                        continue
                    view_people.append((view_name, frame_people, image_path, data_path, camera))

                if len(view_people) < self.min_num_images:
                    drops["too_few_readable_views"] += 1
                    continue

                person_ids = sorted(
                    {pid for _, fp, _, _, _ in view_people for pid in fp},
                    key=lambda item: int(item),
                )
                people_params = {}
                for person_id in person_ids:
                    person = next(fp[person_id] for _, fp, _, _, _ in view_people
                                  if person_id in fp)
                    people_params[person_id] = {
                        "person_key": f"person_{int(person_id):02d}",
                        # raw pyd key, needed to look this person up in a specific
                        # view's pyd for the landmark / contact GT.
                        "pyd_key": person_id,
                        "smpl_pose": np.asarray(person["pose_world"], dtype=np.float32).reshape(-1)[:72],
                        "smpl_beta": np.asarray(person["shape"], dtype=np.float32).reshape(-1)[:10],
                        "smpl_trans": np.asarray(person["trans_world"], dtype=np.float32).reshape(-1)[:3],
                        "gender": person.get("gender", "neutral"),
                        # instance-mask value for this person == person_idx + 1.
                        "person_idx": int(person.get("person_idx", int(person_id))),
                    }

                view_annos = []
                for view_name, frame_people, image_path, data_path, camera in view_people:
                    # Full-length and in a fixed order for EVERY view, so position i
                    # is the same person everywhere.  The old code filtered this list
                    # per view, which both changed its length (hence the equality
                    # guard below dropping whole frames) and silently mis-aligned
                    # positional indexing in get_data.
                    people_annos = [
                        dict(people_params[person_id],
                             visible_in_view=bool(person_id in frame_people))
                        for person_id in person_ids
                    ]
                    if not people_annos:
                        continue

                    # The instance mask ships as .jpg in some scenes and .png in
                    # others (harmony4d vs moyo).  Hard-coding .jpg silently left
                    # mask_path=None for the .png scenes, which then trained the
                    # mask head against an all-zero target for every person.
                    mask_path = None
                    for suffix in (".mask.jpg", ".mask.png", ".mask.jpeg"):
                        candidate = data_path.with_name(f"{frame}{suffix}")
                        if candidate.is_file():
                            mask_path = candidate
                            break
                    view_annos.append(
                        {
                            "view_name": str(view_name),
                            "image_path": str(image_path),
                            "intrinsics": camera["intrinsics"],
                            "extrinsics": camera["extrinsics"],
                            "people": people_annos,
                            "num_people": len(people_annos),
                            "raw_mamma": True,
                            # for on-the-fly landmark / mask GT (loaded lazily in get_data)
                            "data_path": str(data_path),
                            "mask_path": str(mask_path) if mask_path is not None else None,
                        }
                    )

                if len(view_annos) < self.min_num_images:
                    drops["too_few_views_after_parse"] += 1
                    continue
                if len({a["num_people"] for a in view_annos}) != 1:
                    # Should now be impossible (every view gets the same full list);
                    # kept as an assertion-style guard against future regressions.
                    drops["inconsistent_num_people"] += 1
                    continue

                rel = seq_dir.name
                try:
                    rel = str(seq_dir.relative_to(self.data_root))
                except ValueError:
                    pass
                frame_key = f"raw_mamma_{rel}_frame_{frame}"
                data_store[frame_key] = view_annos
                try:
                    numeric_frame_id = self._numeric_frame_id(frame)
                except ValueError:
                    drops["non_numeric_frame_id"] += 1
                    del data_store[frame_key]
                    continue
                sequence_frames[rel].append((numeric_frame_id, str(frame), frame_key))
                kept_here += 1

            per_seq_kept[str(seq_dir)] = kept_here
            if seq_i % 20 == 0 or seq_i == total_seqs:
                logging.info(
                    "SysSMPLMulti: built %d/%d sequences (%d frames so far)",
                    seq_i, total_seqs, len(data_store),
                )

        if bad_pyds:
            logging.warning(
                "SysSMPLMulti: skipped %d unreadable/corrupt pyd file(s) during build (e.g. %s)",
                len(bad_pyds), ", ".join(bad_pyds[:5]),
            )

        total_seen = sum(per_seq_seen.values())
        logging.info(
            "SysSMPLMulti build summary: %d/%d candidate frames kept (%.0f%%) from %d sequence dir(s)",
            len(data_store), total_seen,
            100.0 * len(data_store) / max(total_seen, 1), total_seqs,
        )
        if drops:
            logging.warning(
                "SysSMPLMulti: dropped frames by reason: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(drops.items())),
            )
        for seq_path, seen in sorted(per_seq_seen.items()):
            kept = per_seq_kept.get(seq_path, 0)
            log = logging.warning if kept == 0 else logging.info
            log("SysSMPLMulti:   %4d/%4d frames  %s%s",
                kept, seen, seq_path, "   <-- CONTRIBUTES NOTHING" if kept == 0 else "")

        return data_store, dict(sequence_frames)

    def _parse_gender_label(self, g_raw) -> int:
        if isinstance(g_raw, np.ndarray):
            g_raw = g_raw.reshape(-1)[0] if g_raw.size > 0 else "neutral"
        if isinstance(g_raw, np.generic):
            g_raw = g_raw.item()
        if isinstance(g_raw, (bytes, bytearray)):
            g_raw = g_raw.decode("utf-8", errors="ignore")
        if isinstance(g_raw, (list, tuple)) and len(g_raw) > 0:
            g_raw = g_raw[0]

        if isinstance(g_raw, str):
            text = g_raw.strip().lower()
            if text.startswith("m"):
                return 0
            if text.startswith("f"):
                return 1
            return 2

        try:
            value = int(g_raw)
            if value in (0, 1, 2):
                return value
        except Exception:
            pass

        return 2

    def _load_view(self, anno, emit_mask, emit_pyd):
        """Load a view's RGB image (+ instance mask / GT pyd when needed).

        Returns ``(image, mask, pyd)`` or ``(None, None, None)`` when the view is
        UNUSABLE: the image is corrupt/unreadable, OR (when ``emit_mask``) a present
        mask file is corrupt, OR (when ``emit_pyd``) the landmark/contact ``.data.pyd``
        is corrupt/unreadable. A legitimately absent mask (``mask_path`` is None) is
        fine and yields ``mask=None``. Validating the pyd HERE (and caching it) means
        the per-view landmark/contact read in get_data cannot crash: a bad pyd makes
        the caller swap in another view instead. Lets the loader ride out scattered
        bad-block corruption in the dataset (images, masks, AND pyds).
        """
        image = read_image_cv2(anno["image_path"])
        if image is None:
            return None, None, None
        mask = None
        if emit_mask and anno.get("mask_path"):
            mask = cv2.imread(anno["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:  # mask present but unreadable -> reject this view
                return None, None, None
        pyd = None
        if emit_pyd:
            try:
                pyd = self._load_pickle(anno["data_path"])
            except Exception:  # corrupt/unreadable GT pyd -> reject this view
                return None, None, None
            if not isinstance(pyd, dict) or not pyd:
                return None, None, None
        return image, mask, pyd

    def _select_good_views(self, metadata, order, n_target, emit_mask, emit_pyd):
        """Pick ``n_target`` views (in ``order``) whose image+mask+pyd load OK.

        Caches the decoded image/mask/pyd so the caller does not re-read them. Returns
        ``(ids, annos, images, masks, pyds)`` or ``None`` if fewer than ``n_target``
        usable views exist (signals the caller to resample a different sample).
        ``order`` must cover every view index so corrupt ones can be replaced.
        """
        chosen, annos, imgs, masks, pyds = [], [], [], [], []
        for vi in order:
            if len(annos) >= n_target:
                break
            img, msk, pyd = self._load_view(metadata[int(vi)], emit_mask, emit_pyd)
            if img is None:
                self._bad_view_reads = getattr(self, "_bad_view_reads", 0) + 1
                n = self._bad_view_reads
                if n in (1, 10, 100) or n % 500 == 0:
                    logging.warning(
                        "SysSMPLMulti: skipped %d corrupt image/mask/pyd view(s) so far "
                        "(e.g. %s); swapping in another view.",
                        n, metadata[int(vi)]["image_path"],
                    )
                continue
            chosen.append(int(vi)); annos.append(metadata[int(vi)])
            imgs.append(img); masks.append(msk); pyds.append(pyd)

        if len(annos) < n_target:
            if self.allow_duplicate_img and annos:
                good = list(zip(chosen, annos, imgs, masks, pyds))
                k = 0
                while len(annos) < n_target:
                    c, a, i, m, p = good[k % len(good)]; k += 1
                    chosen.append(c); annos.append(a); imgs.append(i); masks.append(m); pyds.append(p)
            else:
                return None
        return chosen, annos, imgs, masks, pyds

    def get_data(
        self,
        seq_index: Optional[int] = None,
        img_per_seq: Optional[int] = None,
        seq_name: Optional[str] = None,
        ids: Optional[List[int]] = None,
        aspect_ratio: float = 1.0,
        _resample_depth: int = 0,
    ) -> dict:
        if not self.use_temporal_training:
            timings = defaultdict(float) if self.profile_data_loading else None
            total_start = time.perf_counter()
            batch = self._get_single_frame_data(
                seq_index=seq_index,
                img_per_seq=img_per_seq,
                seq_name=seq_name,
                ids=ids,
                aspect_ratio=aspect_ratio,
                _resample_depth=_resample_depth,
                _profile_timings=timings,
            )
            if timings is not None:
                timings["dataset_total"] += time.perf_counter() - total_start
                batch["_profile_timings"] = dict(timings)
            return batch

        total_start = time.perf_counter()
        timings = defaultdict(float) if self.profile_data_loading else None
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        clip = self.data_store[seq_name]

        requested = int(img_per_seq or len(clip["common_view_names"]))
        common_views = list(clip["common_view_names"])
        if requested > len(common_views):
            return self._resample_temporal_clip(
                img_per_seq, aspect_ratio, _resample_depth,
                reason=f"only {len(common_views)} shared views",
            )
        if self.fixed_view_sampling:
            selected_view_names = common_views[:requested]
        else:
            order = np.random.permutation(len(common_views))[:requested]
            selected_view_names = [common_views[int(i)] for i in order]

        frame_batches = []
        for frame_key in clip["frame_keys"]:
            frame_batch = self._get_single_frame_data(
                seq_name=frame_key,
                img_per_seq=requested,
                aspect_ratio=aspect_ratio,
                required_view_names=selected_view_names,
                _profile_timings=timings,
            )
            if frame_batch is None:
                return self._resample_temporal_clip(
                    img_per_seq, aspect_ratio, _resample_depth,
                    reason="a selected view was unreadable",
                )
            frame_batches.append(frame_batch)

        combine_start = time.perf_counter()
        batch = self._combine_temporal_frame_batches(
            seq_name=seq_name,
            clip=clip,
            frame_batches=frame_batches,
            selected_view_names=selected_view_names,
        )
        if timings is not None:
            timings["temporal_combine"] += time.perf_counter() - combine_start
            timings["dataset_total"] += time.perf_counter() - total_start
            batch["_profile_timings"] = dict(timings)
        return batch

    def _resample_temporal_clip(self, img_per_seq, aspect_ratio, depth, reason):
        if depth < 20 and self.sequence_list_len > 1:
            logging.warning(
                "SysSMPLMulti: temporal clip unusable (%s); resampling.", reason
            )
            return self.get_data(
                seq_index=random.randint(0, self.sequence_list_len - 1),
                img_per_seq=img_per_seq,
                aspect_ratio=aspect_ratio,
                _resample_depth=depth + 1,
            )
        raise RuntimeError(
            f"Could not gather a valid temporal clip after {depth} resamples: {reason}"
        )

    def _combine_temporal_frame_batches(
        self, *, seq_name, clip, frame_batches, selected_view_names
    ):
        """Stack single-frame batches into a time-major T*V sample."""
        T = len(frame_batches)
        V = len(selected_view_names)
        union_person_keys = sorted(
            {key for frame_batch in frame_batches for key in frame_batch["person_keys"]}
        )[: int(self.max_num_people)]
        dst_person = {key: idx for idx, key in enumerate(union_person_keys)}
        P = int(self.max_num_people)

        person_keys = (
            "smpl_pose", "smpl_beta", "smpl_trans", "smpl_gender", "has_smpl"
        )
        view_person_keys = (
            "smpl_joints2d", "smpl_joints3d_world", "smpl_joints2d_confidence",
            "smpl_landmarks2d", "smpl_landmarks2d_visibility", "person_mask",
            "smpl_contact", "smpl_floor_contact",
        )
        view_keys = (
            "images", "depths", "extrinsics", "intrinsics", "cam_points",
            "world_points", "point_masks", "original_sizes", "image_paths",
        )

        combined = {
            "seq_name": "syssmpl_multi_" + seq_name,
            "frame_num": T * V,
            "temporal_num_frames": T,
            "views_per_frame": V,
            "frame_ids": np.asarray(clip["frame_ids"], dtype=np.int64),
            "view_ids": np.arange(V, dtype=np.int64),
            "ids": np.tile(np.arange(V, dtype=np.int64), T),
            "person_keys": union_person_keys,
            "num_people": np.asarray(
                [len(frame_batch["person_keys"]) for frame_batch in frame_batches],
                dtype=np.int64,
            ),
        }

        for key in view_keys:
            if all(key in frame_batch for frame_batch in frame_batches):
                values = []
                for frame_batch in frame_batches:
                    values.extend(list(frame_batch[key]))
                combined[key] = values

        for key in person_keys:
            if not all(key in frame_batch for frame_batch in frame_batches):
                continue
            sample = np.asarray(frame_batches[0][key])
            fill = 2 if key == "smpl_gender" else 0
            out = np.full((T, P, *sample.shape[1:]), fill, dtype=sample.dtype)
            for t, frame_batch in enumerate(frame_batches):
                for src_idx, person_key in enumerate(frame_batch["person_keys"]):
                    if person_key in dst_person:
                        out[t, dst_person[person_key]] = np.asarray(frame_batch[key])[src_idx]
            combined[key] = out

        for key in view_person_keys:
            if not all(key in frame_batch for frame_batch in frame_batches):
                continue
            sample = np.asarray(frame_batches[0][key])
            out = np.zeros((T * V, P, *sample.shape[2:]), dtype=sample.dtype)
            for t, frame_batch in enumerate(frame_batches):
                value = np.asarray(frame_batch[key])
                for src_idx, person_key in enumerate(frame_batch["person_keys"]):
                    if person_key in dst_person:
                        out[t * V : (t + 1) * V, dst_person[person_key]] = value[:, src_idx]
            combined[key] = out

        combined["selected_view_names"] = list(selected_view_names)
        return combined

    def _get_single_frame_data(
        self,
        seq_index: Optional[int] = None,
        img_per_seq: Optional[int] = None,
        seq_name: Optional[str] = None,
        ids: Optional[List[int]] = None,
        aspect_ratio: float = 1.0,
        _resample_depth: int = 0,
        required_view_names: Optional[List[str]] = None,
        _profile_timings=None,
    ) -> dict:
        frame_start = time.perf_counter()
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.frame_data_store[seq_name]
        n_views = len(metadata)

        # Build the full ordered list of view indices to TRY (corrupt ones get
        # skipped and replaced by later entries), plus the target view count.
        if required_view_names is not None:
            by_name = {str(anno["view_name"]): idx for idx, anno in enumerate(metadata)}
            if any(name not in by_name for name in required_view_names):
                return None
            view_order = [by_name[name] for name in required_view_names]
            n_target = len(required_view_names)
        elif ids is None:
            if self.fixed_view_sampling:
                requested = img_per_seq or n_views
                if requested > n_views:
                    logging.warning(
                        "Requested %d views but only %d available in %s; clamping to available views.",
                        requested, n_views, seq_name,
                    )
                    requested = n_views
                n_target = requested
                view_order = list(range(n_views))                     # deterministic
            else:
                n_target = int(img_per_seq)
                view_order = list(np.random.permutation(n_views))     # random, full cover
        else:
            n_target = len(ids)
            seen = {int(i) for i in ids}
            view_order = [int(i) for i in ids] + [i for i in range(n_views) if i not in seen]

        _is_raw_sel = bool(metadata[0].get("raw_mamma", False))
        emit_mask_sel = bool(self.emit_person_mask and _is_raw_sel)
        # pyd is (re-)read for landmark/contact GT in the loop below; validate+cache
        # it here so a corrupt pyd swaps the view instead of crashing mid-loop.
        emit_pyd_sel = bool((self.emit_landmarks or self.emit_contact)
                            and self._verts512 is not None and _is_raw_sel)
        read_start = time.perf_counter()
        sel = self._select_good_views(metadata, view_order, n_target, emit_mask_sel, emit_pyd_sel)
        if _profile_timings is not None:
            _profile_timings["view_read"] += time.perf_counter() - read_start
        if sel is None:
            if required_view_names is not None:
                return None
            # Not enough readable views in this frame -> resample a different sample.
            if _resample_depth < 20 and self.sequence_list_len > 1:
                logging.warning(
                    "SysSMPLMulti: %s has < %d readable views; resampling another sample.",
                    seq_name, n_target,
                )
                new_idx = random.randint(0, self.sequence_list_len - 1)
                return self._get_single_frame_data(
                    seq_index=new_idx, img_per_seq=img_per_seq,
                    aspect_ratio=aspect_ratio, _resample_depth=_resample_depth + 1,
                )
            raise RuntimeError(
                f"SysSMPLMulti: could not gather {n_target} readable views for {seq_name} "
                f"after {_resample_depth} resamples (dataset too corrupt?)."
            )
        ids, annos, view_images, view_masks, view_pyds = sel
        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []
        person_count = metadata[0]["num_people"]
        padded_people = int(self.max_num_people)
        person_count = min(person_count, padded_people)
        person_anchor = metadata[0]["people"][:person_count]

        smpl_joints2d_list = []
        smpl_joints3d_world_list = []
        confidences = []

        # Dense-landmark / mask GT are only available for raw Mamma_mv_split
        # (the pyd ships vertices2d / vertex_visibility / *.mask.jpg).
        is_raw = bool(metadata[0].get("raw_mamma", False))
        emit_lmk = self.emit_landmarks and self._verts512 is not None and is_raw
        emit_mask = self.emit_person_mask and is_raw
        emit_ct = self.emit_contact and self._verts512 is not None and is_raw
        landmarks2d_list = []
        landmarks_vis_list = []
        person_mask_list = []
        contact_list = []
        floor_contact_list = []

        smpl_poses = np.zeros((padded_people, 72), dtype=np.float32)
        smpl_betas = np.zeros((padded_people, 10), dtype=np.float32)
        smpl_translations = np.zeros((padded_people, 3), dtype=np.float32)
        smpl_genders = np.full((padded_people,), 2, dtype=np.int64)
        has_smpl = np.zeros((padded_people,), dtype=np.float32)
        person_keys = []
        # World joints are camera-independent: batch-decode each person once per
        # frame, then only run the inexpensive projection for individual views.
        decode_start = time.perf_counter()
        joints_world_by_person = self._decode_raw_mamma_joints_world_batch(person_anchor)
        if _profile_timings is not None:
            _profile_timings["smpl_decode"] += time.perf_counter() - decode_start

        for person_idx, person in enumerate(person_anchor):
            smpl_poses[person_idx] = np.asarray(person["smpl_pose"], dtype=np.float32).reshape(-1)[:72]
            smpl_betas[person_idx] = np.asarray(person["smpl_beta"], dtype=np.float32).reshape(-1)[:10]
            if "smpl_trans" in person:
                smpl_translations[person_idx] = np.asarray(person["smpl_trans"], dtype=np.float32).reshape(-1)[:3]
            smpl_genders[person_idx] = self._parse_gender_label(person.get("gender", "neutral"))
            has_smpl[person_idx] = 1.0
            person_keys.append(person.get("person_key", f"person_{person_idx}"))

        view_loop_start = time.perf_counter()
        preprocess_elapsed = 0.0
        for view_i, anno in enumerate(annos):
            image_path = anno["image_path"]
            image = view_images[view_i]   # pre-loaded & validated in _select_good_views
            depth_map = (
                np.zeros(image.shape[:2], dtype=np.float32)
                if self.emit_dense_geometry
                else None
            )

            extri_opencv = np.copy(anno["extrinsics"])
            intri_opencv = np.copy(anno["intrinsics"])
            original_size = np.asarray(
                anno.get("original_size", image.shape[:2]), dtype=np.int64
            ).copy()

            people = anno["people"][:person_count]
            joints3d_world = np.zeros((padded_people, 24, 3), dtype=np.float32)
            joints2d_orig = np.zeros((padded_people, 24, 2), dtype=np.float32)
            for person_idx, person in enumerate(people):
                # Raw Mamma_mv_split .data.pyd stores vertices3d/joints3d in camera
                # coords with a high-rank joints2d tensor, so recreate the training
                # joint targets from the SMPL-X world params (same convention as the
                # loss) and project them with this view's camera.
                joints_world = joints_world_by_person[person_idx]
                joints3d_world[person_idx] = joints_world[:24]
                joints2d_orig[person_idx] = self._project_points_opencv_np(
                    joints_world[:24],
                    extri_opencv,
                    intri_opencv,
                )
                if anno.get("preprocessed_518", False):
                    joints2d_orig[person_idx] += np.asarray(
                        anno["track_offset"], dtype=np.float32
                    )

            # --- optional dense-landmark GT (raw only): M @ vertices2d in ORIG
            # pixels, appended to the track so it gets the same crop/resize as
            # the joints; visibility from M @ vertex_visibility (per view). ---
            landmarks2d_orig = None
            landmarks_vis = None
            contact_gt = None
            floor_contact_gt = None
            if emit_lmk or emit_ct:
                view_pyd = view_pyds[view_i]   # pre-loaded & validated in _select_good_views
                # (pyd_ids no longer used: people are looked up by pyd_key below)
            if emit_lmk:
                landmarks2d_orig = np.zeros((padded_people, 512, 2), dtype=np.float32)
                landmarks_vis = np.zeros((padded_people, 512), dtype=np.float32)
                for person_idx in range(person_count):
                    # Look the person up by pyd key, never positionally: a person can
                    # be absent from THIS view's pyd (occlusion) while still being a
                    # real person in the frame, so pyd_ids[person_idx] would silently
                    # be a different person -- or out of range.
                    key = people[person_idx].get("pyd_key")
                    if key is None or key not in view_pyd:
                        continue          # visibility stays 0 -> unsupervised in this view
                    p = view_pyd[key]
                    v2d = np.asarray(p["vertices2d"], dtype=np.float32)      # (10475,2)
                    landmarks2d_orig[person_idx] = downsample_vertices(self._verts512, v2d)
                    vv = np.asarray(p["vertex_visibility"], dtype=np.float32).reshape(-1)
                    landmarks_vis[person_idx] = downsample_visibility(
                        self._verts512, vv[None, :],
                        threshold=self.landmark_visibility_threshold,
                    )[0]
            # --- optional MAMMA-style contact GT (raw only), per (view, person) ---
            #   floor_contact = down( floor_contact_mask )
            #   contact       = down( sdf_vertices < thresh ) * (1 - visible)
            # (MAMMA: a visible landmark is treated as NOT in contact.)
            if emit_ct:
                contact_gt = np.zeros((padded_people, 512), dtype=np.float32)
                floor_contact_gt = np.zeros((padded_people, 512), dtype=np.float32)
                for person_idx in range(person_count):
                    key = people[person_idx].get("pyd_key")
                    if key is None or key not in view_pyd:
                        contact_gt[person_idx] = -1.0        # absent here -> ignore
                        floor_contact_gt[person_idx] = -1.0
                        continue
                    p = view_pyd[key]
                    vis512 = downsample_visibility(
                        self._verts512,
                        np.asarray(p["vertex_visibility"], dtype=np.float32).reshape(1, -1),
                        threshold=self.landmark_visibility_threshold,
                    )[0]
                    # person-person contact via SDF. In this dataset sdf_vertices is
                    # only exported for SOME samples; when absent, mark the whole row
                    # invalid (-1) so the loss ignores it (floor contact is always present).
                    sdf = np.asarray(p.get("sdf_vertices", []), dtype=np.float32).reshape(-1)
                    if sdf.size == NUM_SMPLX_VERTS:
                        contact_v = (sdf < self.contact_threshold).astype(np.float32)
                        c512 = downsample_visibility(self._verts512, contact_v[None, :], threshold=0.5)[0]
                        contact_gt[person_idx] = c512 * (1.0 - vis512)
                    else:
                        # no SDF: MAMMA treats as no-contact (0, negatives); otherwise -1 = ignore.
                        contact_gt[person_idx] = 0.0 if self.contact_missing_as_negative else -1.0
                    floor_v = np.asarray(p.get("floor_contact_mask", []), dtype=np.float32).reshape(-1)
                    if floor_v.size == NUM_SMPLX_VERTS:
                        floor_contact_gt[person_idx] = downsample_visibility(
                            self._verts512, floor_v[None, :], threshold=0.5
                        )[0]
                    else:
                        floor_contact_gt[person_idx] = -1.0

            # --- optional per-person instance mask (raw only): transformed with
            # the image via extra_maps (nearest interp preserves labels). ---
            extra_maps = None
            if emit_mask and anno.get("mask_path"):
                instance_mask = view_masks[view_i]   # pre-loaded & validated (not None here)
                if instance_mask is not None:
                    extra_maps = {"person_mask": instance_mask.astype(np.float32)}

            n_joint_pts = padded_people * 24
            if landmarks2d_orig is not None:
                track_in = np.concatenate(
                    [joints2d_orig.reshape(-1, 2), landmarks2d_orig.reshape(-1, 2)], axis=0
                )
            else:
                track_in = joints2d_orig.reshape(-1, 2)

            preprocess_start = time.perf_counter()
            if anno.get("preprocessed_518", False):
                # RGB, mask and K were transformed together by the offline composer.
                # Repeating process_one_image here would crop/resize a second time and
                # corrupt both K and 2D GT. Photometric augmentation is applied later
                # by ComposedDataset, so skipping this geometry does not disable jitter.
                if tuple(image.shape[:2]) != tuple(int(x) for x in target_image_shape):
                    raise ValueError(
                        f"Cached image {image_path} has shape {image.shape[:2]}, but "
                        f"the requested target is {tuple(target_image_shape)}."
                    )
                if self.emit_dense_geometry:
                    raise ValueError(
                        "preprocessed_518 fast path requires emit_dense_geometry=false"
                    )
                world_coords_points = cam_coords_points = point_mask = None
                track_new = np.asarray(track_in, dtype=np.float32).copy()
                x, y = track_new[..., 0], track_new[..., 1]
                confidence = (
                    (x >= 0)
                    & (x < image.shape[1])
                    & (y >= 0)
                    & (y < image.shape[0])
                ).astype(np.float32)
            else:
                (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    track_new,
                    confidence,
                ) = self.process_one_image(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    original_size,
                    target_image_shape,
                    track=track_in,
                    filepath=image_path,
                    extra_maps=extra_maps,
                    profile_timings=_profile_timings,
                )
            preprocess_elapsed += time.perf_counter() - preprocess_start
            H_final, W_final = image.shape[:2]
            joints2d_new = track_new[:n_joint_pts].reshape(padded_people, 24, 2)
            if confidence is not None:
                joints_conf = confidence[:n_joint_pts].reshape(padded_people, 24)
                # process_one_image's confidence is only an IN-FRAME test, so a person
                # occluded in THIS view still scores 1.  Zero those: their 3D
                # supervision is unaffected, only this view's 2D reprojection is
                # skipped -- which is the whole point of a multi-view setup.
                for person_idx in range(person_count):
                    if not people[person_idx].get("visible_in_view", True):
                        joints_conf[person_idx] = 0.0
            else:
                joints_conf = None

            if landmarks2d_orig is not None:
                lmk_px = track_new[n_joint_pts:].reshape(padded_people, 512, 2)
                # normalise to [0, 1] to match the head's sigmoid output convention.
                lmk_norm = np.empty_like(lmk_px)
                lmk_norm[..., 0] = lmk_px[..., 0] / W_final
                lmk_norm[..., 1] = lmk_px[..., 1] / H_final
                # gate visibility by in-frame after the geometric transform.
                if confidence is not None:
                    inframe = confidence[n_joint_pts:].reshape(padded_people, 512)
                    landmarks_vis = landmarks_vis * inframe
                landmarks2d_list.append(lmk_norm.astype(np.float32))
                landmarks_vis_list.append(landmarks_vis.astype(np.float32))

            if emit_ct:
                # contact GT is per-landmark (not a pixel coord) -> no crop/resize.
                contact_list.append(contact_gt.astype(np.float32))
                floor_contact_list.append(floor_contact_gt.astype(np.float32))

            if emit_mask:
                # Match the model's ACTUAL mask-head grid, which follows the
                # processed image size (the dynamic sampler may use a non-square
                # aspect ratio, so it is NOT always img_size//stride).
                #   stride = patch_size (legacy, default) -> the dot-product head's
                #     patch grid (37x37 for 518/14);
                #   stride = person_mask_stride (e.g. 2)  -> the DPT head's
                #     pixel-level grid (259x259 for 518).
                mask_stride = self.person_mask_stride or self.patch_size
                mask_h_final = H_final // mask_stride
                mask_w_final = W_final // mask_stride
                person_mask = np.zeros(
                    (padded_people, mask_h_final, mask_w_final), dtype=np.float32
                )
                if extra_maps is not None and extra_maps.get("person_mask") is not None:
                    mask_final = extra_maps["person_mask"]
                    for person_idx in range(person_count):
                        pv = int(people[person_idx].get("person_idx", person_idx)) + 1
                        person_mask[person_idx] = rasterize_person_patch_mask(
                            mask_final, pv, mask_h_final, mask_w_final
                        )
                person_mask_list.append(person_mask)

            images.append(image)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            if depth_map is not None:
                depths.append(depth_map)
            if cam_coords_points is not None:
                cam_points.append(cam_coords_points)
            if world_coords_points is not None:
                world_points.append(world_coords_points)
            if point_mask is not None:
                point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

            smpl_joints3d_world_list.append(joints3d_world)
            smpl_joints2d_list.append(joints2d_new)
            confidences.append(joints_conf)

        view_loop_elapsed = time.perf_counter() - view_loop_start
        if _profile_timings is not None:
            _profile_timings["crop_resize"] += preprocess_elapsed
            _profile_timings["view_other"] += max(
                0.0, view_loop_elapsed - preprocess_elapsed
            )

        pack_start = time.perf_counter()
        smpl_joints2d = np.stack(smpl_joints2d_list, axis=0).astype(np.float32)
        smpl_joints3d_world = np.stack(smpl_joints3d_world_list, axis=0).astype(np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)

        batch = {
            "seq_name": "syssmpl_multi_" + seq_name,
            "ids": np.asarray(ids, dtype=np.int64),
            "frame_num": len(extrinsics),
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "original_sizes": original_sizes,
            "smpl_pose": smpl_poses,
            "smpl_beta": smpl_betas,
            "smpl_trans": smpl_translations,
            "smpl_joints2d": smpl_joints2d,
            "smpl_joints3d_world": smpl_joints3d_world,
            "smpl_gender": smpl_genders,
            "has_smpl": has_smpl,
            "num_people": np.asarray(person_count, dtype=np.int64),
            "person_keys": person_keys,
            "smpl_joints2d_confidence": confidences,
            "image_paths": image_paths,
        }
        if depths:
            batch["depths"] = depths
        if cam_points:
            batch["cam_points"] = cam_points
        if world_points:
            batch["world_points"] = world_points
        if point_masks:
            batch["point_masks"] = point_masks

        # Dense-landmark GT: (S, P, 512, 2) normalised 2D + (S, P, 512) visibility.
        if landmarks2d_list:
            batch["smpl_landmarks2d"] = np.stack(landmarks2d_list, axis=0).astype(np.float32)
            batch["smpl_landmarks2d_visibility"] = np.stack(landmarks_vis_list, axis=0).astype(np.float32)
        # Per-person mask GT: (S, P, patch_grid, patch_grid) occupancy in [0,1].
        if person_mask_list:
            batch["person_mask"] = np.stack(person_mask_list, axis=0).astype(np.float32)
        # MAMMA-style contact GT: (S, P, 512) binary person-person + floor contact.
        if contact_list:
            batch["smpl_contact"] = np.stack(contact_list, axis=0).astype(np.float32)
            batch["smpl_floor_contact"] = np.stack(floor_contact_list, axis=0).astype(np.float32)

        if _profile_timings is not None:
            _profile_timings["frame_pack"] += time.perf_counter() - pack_start
            _profile_timings["frame_total"] += time.perf_counter() - frame_start
        return batch
