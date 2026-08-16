"""Harmony4D multi-person, multi-view training dataset.

One sample is one annotated time step observed by ``img_per_seq`` exo cameras.
Harmony4D stores standard neutral-SMPL fits in an Aria/world coordinate frame,
while its GoPro calibration is a COLMAP ``OPENCV_FISHEYE`` reconstruction.  The
loader therefore converts COLMAP world-to-camera poses into the SMPL world,
removes the similarity scale from their rotation blocks, and undistorts RGB to
a pinhole image before calling :class:`BaseDataset`'s crop/resize pipeline.

Expected sequence layout::

    <root>/<activity>/<sequence>/
      exo/camXX/images/<frame>.jpg
      colmap/workplace/{cameras.txt,images.txt,scale.npy}
      # ego/exo sequences use aria_from_colmap_transforms.pkl instead of scale.npy
      processed_data/smpl/<frame>.npy

The emitted tensors follow the same batch contract as SysSMPLMultiDataset. By
default they use Harmony4D's classic-SMPL annotations; ``body_model_type=smplx``
loads offline body-only SMPL-X fits. When ``emit_person_mask=True``, only
complete rectified RGB/mask/camera bundles are indexed and synchronized
per-person mask targets are emitted. Dense landmark and contact targets are not
emitted.
"""

from __future__ import annotations

import logging
import pickle
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import read_image_cv2
from training.data.landmark_mask_gt import rasterize_person_patch_mask


@dataclass(frozen=True)
class _CameraCalibration:
    name: str
    image_dir: Path
    width: int
    height: int
    model: str
    params: np.ndarray
    extrinsics: np.ndarray


@dataclass
class _SequenceRecord:
    name: str
    path: Path
    frames: List[str]
    cameras: List[_CameraCalibration]


class Harmony4DDataset(BaseDataset):
    """Load synchronized Harmony4D exo views with SMPL/SMPL-X supervision."""

    PEOPLE = ("aria01", "aria02")

    def __init__(
        self,
        common_conf,
        split: str = "train",
        Harmony4D_DIR: str | None = None,
        min_num_images: int = 8,
        max_num_people: int = 2,
        val_sequence_fraction: float = 0.1,
        split_seed: int = 42,
        max_sequences: Optional[int] = None,
        max_frames_per_sequence: Optional[int] = None,
        # Downscale the 4K fisheye image before remapping. This is geometrically
        # equivalent after K is scaled, but drastically reduces map RAM and I/O.
        undistort_max_side: Optional[int] = 1280,
        undistort_balance: float = 0.0,
        undistort_cache_size: int = 8,
        fixed_anchor_camera: Optional[str] = None,
        body_model_type: str = "smpl",
        smplx_annotation_root: Optional[str] = None,
        person_mask_root: Optional[str] = None,
        emit_person_mask: bool = False,
        person_mask_stride: Optional[int] = None,
    ):
        super().__init__(common_conf=common_conf)

        if Harmony4D_DIR is None:
            raise ValueError("Harmony4D_DIR must be specified")
        self.root = Path(Harmony4D_DIR).expanduser()
        if not self.root.is_dir():
            raise ValueError(f"Harmony4D root not found: {self.root}")

        self.split = str(split).lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if not 0.0 <= float(val_sequence_fraction) < 1.0:
            raise ValueError("val_sequence_fraction must be in [0, 1)")

        self.training = bool(common_conf.training)
        self.fixed_view_sampling = bool(
            getattr(common_conf, "fixed_view_sampling", False)
        )
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
        self.min_num_images = int(min_num_images)
        self.max_num_people = int(max_num_people)
        if self.max_num_people < len(self.PEOPLE):
            raise ValueError(
                f"Harmony4D has {len(self.PEOPLE)} people; max_num_people="
                f"{self.max_num_people} would truncate ground truth"
            )
        self.body_model_type = str(body_model_type).strip().lower()
        if self.body_model_type not in {"smpl", "smplx"}:
            raise ValueError("body_model_type must be 'smpl' or 'smplx'")
        self.smplx_annotation_root = (
            None
            if smplx_annotation_root is None
            else Path(smplx_annotation_root).expanduser()
        )
        if self.body_model_type == "smplx" and (
            self.smplx_annotation_root is None
            or not self.smplx_annotation_root.is_dir()
        ):
            raise ValueError(
                "body_model_type=smplx requires an existing "
                f"smplx_annotation_root, got {self.smplx_annotation_root}"
            )
        self.emit_person_mask = bool(emit_person_mask)
        self.person_mask_root = (
            None
            if person_mask_root is None
            else Path(person_mask_root).expanduser()
        )
        if self.emit_person_mask and (
            self.person_mask_root is None or not self.person_mask_root.is_dir()
        ):
            raise ValueError(
                "emit_person_mask=True requires an existing person_mask_root, "
                f"got {self.person_mask_root}"
            )
        self.person_mask_stride = (
            int(person_mask_stride)
            if person_mask_stride is not None
            else None
        )
        if self.person_mask_stride is not None and self.person_mask_stride < 1:
            raise ValueError("person_mask_stride must be positive")

        self.undistort_max_side = (
            None if undistort_max_side is None else int(undistort_max_side)
        )
        self.undistort_balance = float(undistort_balance)
        self.undistort_cache_size = max(0, int(undistort_cache_size))
        self.fixed_anchor_camera = (
            str(fixed_anchor_camera) if fixed_anchor_camera else None
        )
        self._undistort_cache: OrderedDict[
            Tuple[str, str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()

        sequence_paths = self._discover_sequences(self.root)
        sequence_paths = self._split_sequences(
            sequence_paths,
            split=self.split,
            val_fraction=float(val_sequence_fraction),
            seed=int(split_seed),
        )
        if max_sequences is not None:
            sequence_paths = sequence_paths[: max(0, int(max_sequences))]

        self.sequences: List[_SequenceRecord] = []
        self.samples: List[Tuple[int, str]] = []
        candidate_frames = 0
        missing_annotations = 0
        incomplete_mask_frames = 0
        for path in sequence_paths:
            try:
                record = self._build_sequence_record(
                    path, max_frames=max_frames_per_sequence
                )
                candidate_frames += len(record.frames)
                if self.body_model_type == "smplx":
                    before = len(record.frames)
                    record.frames = [
                        frame
                        for frame in record.frames
                        if self._annotation_path(record.path, frame).is_file()
                    ]
                    missing_annotations += before - len(record.frames)
                if self.emit_person_mask:
                    # Filter incomplete pseudo-GT before samples reach the sampler.
                    # A frame is usable only when at least min_num_images cameras
                    # have the full rectified RGB + mask + camera-metadata bundle.
                    before = len(record.frames)
                    record.frames = [
                        frame
                        for frame in record.frames
                        if sum(
                            self._view_available(record, camera, frame)
                            for camera in record.cameras
                        )
                        >= self.min_num_images
                    ]
                    incomplete_mask_frames += before - len(record.frames)
            except Exception as exc:
                logging.warning("Harmony4D: skipping %s: %s", path, exc)
                continue
            if len(record.cameras) < self.min_num_images or not record.frames:
                continue
            seq_idx = len(self.sequences)
            self.sequences.append(record)
            self.samples.extend((seq_idx, frame) for frame in record.frames)

        if not self.samples:
            raise ValueError(
                f"No usable Harmony4D samples under {self.root} for split={self.split}"
            )

        self.temporal_samples = (
            self._build_temporal_samples() if self.use_temporal_training else []
        )
        self.len_train = (
            len(self.temporal_samples)
            if self.use_temporal_training
            else len(self.samples)
        )
        if self.use_temporal_training and not self.temporal_samples:
            raise ValueError(
                "Harmony4D temporal training produced no consecutive clips; "
                f"T={self.temporal_clip_length}, stride={self.temporal_clip_stride}"
            )

        logging.info(
            "Harmony4D %s: %d sequences, %d timesteps, %d-view minimum, root=%s",
            self.split,
            len(self.sequences),
            len(self.samples),
            self.min_num_images,
            self.root,
        )
        if self.use_temporal_training:
            logging.info(
                "Harmony4D %s temporal clips: %d (T=%d, stride=%d)",
                self.split,
                len(self.temporal_samples),
                self.temporal_clip_length,
                self.temporal_clip_stride,
            )
        logging.info(
            "Harmony4D %s filtering: candidates=%d, missing_annotations=%d, "
            "incomplete_mask_frames=%d, retained=%d",
            self.split,
            candidate_frames,
            missing_annotations,
            incomplete_mask_frames,
            len(self.samples),
        )

    @staticmethod
    def _numeric_frame_id(frame: str) -> int:
        matches = re.findall(r"\d+", str(frame))
        if not matches:
            raise ValueError(f"Frame name has no numeric id: {frame!r}")
        return int(matches[-1])

    def _build_temporal_samples(self):
        clips = []
        T = self.temporal_clip_length
        stride = self.temporal_clip_stride
        for sequence_index, record in enumerate(self.sequences):
            ordered = sorted(
                record.frames, key=lambda frame: self._numeric_frame_id(frame)
            )
            runs = []
            for frame in ordered:
                frame_id = self._numeric_frame_id(frame)
                if not runs or frame_id - runs[-1][-1][0] != 1:
                    runs.append([])
                runs[-1].append((frame_id, frame))
            for run in runs:
                if len(run) < T:
                    continue
                for start in range(0, len(run) - T + 1, stride):
                    window = run[start : start + T]
                    clips.append(
                        (
                            sequence_index,
                            tuple(frame for _, frame in window),
                            tuple(frame_id for frame_id, _ in window),
                        )
                    )
        return clips

    @staticmethod
    def _discover_sequences(root: Path) -> List[Path]:
        # Accept the dataset root, one activity directory, or one sequence.
        # The first form is used by training; the narrower forms are convenient
        # for deterministic inspection and calibration smoke tests.
        if (root / "processed_data" / "smpl").is_dir():
            return [root]
        activity_sequences = sorted(
            smpl_dir.parent.parent
            for smpl_dir in root.glob("*/processed_data/smpl")
            if smpl_dir.is_dir()
        )
        if activity_sequences:
            return activity_sequences
        return sorted(
            smpl_dir.parent.parent
            for smpl_dir in root.glob("*/*/processed_data/smpl")
            if smpl_dir.is_dir()
        )

    @staticmethod
    def _split_sequences(
        paths: List[Path], split: str, val_fraction: float, seed: int
    ) -> List[Path]:
        """Make a deterministic sequence-level split (never split adjacent frames)."""
        if val_fraction <= 0.0 or len(paths) <= 1:
            return paths if split == "train" else []
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(paths))
        n_val = max(1, int(round(len(paths) * val_fraction)))
        val_ids = set(int(i) for i in order[:n_val])
        if split == "train":
            return [path for i, path in enumerate(paths) if i not in val_ids]
        return [path for i, path in enumerate(paths) if i in val_ids]

    @staticmethod
    def _quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
        q = np.asarray(qvec, dtype=np.float64).reshape(4)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        w, x, y, z = q
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _orthonormalize_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
        """Remove the uniform similarity scale while preserving projections."""
        extr = np.asarray(extrinsics, dtype=np.float64).copy()
        det = float(np.linalg.det(extr[:3, :3]))
        scale = np.sign(det) * abs(det) ** (1.0 / 3.0)
        if abs(scale) < 1e-9:
            raise ValueError("degenerate camera rotation")
        extr[:3, :4] /= scale

        # Similarity inputs should already be extremely close to SO(3). SVD only
        # removes tiny numeric drift; translation must not be changed afterwards.
        u, _, vt = np.linalg.svd(extr[:3, :3])
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        extr[:3, :3] = rotation
        return extr[:3].astype(np.float32)

    @staticmethod
    def _read_colmap_cameras(path: Path) -> Dict[int, Tuple[str, int, int, np.ndarray]]:
        cameras = {}
        with path.open("r") as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                camera_id = int(fields[0])
                cameras[camera_id] = (
                    fields[1],
                    int(fields[2]),
                    int(fields[3]),
                    np.asarray(fields[4:], dtype=np.float64),
                )
        return cameras

    @staticmethod
    def _read_colmap_exo_poses(
        path: Path,
    ) -> Dict[str, Tuple[str, np.ndarray, np.ndarray, int]]:
        """Use the earliest SfM image pose for each static exo camera."""
        candidates: Dict[str, List[Tuple[str, np.ndarray, np.ndarray, int]]] = {}
        with path.open("r") as stream:
            for line in stream:
                fields = line.split()
                if len(fields) != 10 or fields[0].startswith("#"):
                    continue
                image_name = fields[9]
                parts = Path(image_name).parts
                if not parts or not parts[0].startswith("cam"):
                    continue
                camera_name = parts[0]
                frame = Path(image_name).stem
                candidates.setdefault(camera_name, []).append(
                    (
                        frame,
                        np.asarray(fields[1:5], dtype=np.float64),
                        np.asarray(fields[5:8], dtype=np.float64),
                        int(fields[8]),
                    )
                )
        return {
            camera: min(values, key=lambda value: int(value[0]))
            for camera, values in candidates.items()
        }

    @staticmethod
    def _world_from_colmap(workplace: Path) -> np.ndarray:
        scale_path = workplace / "scale.npy"
        if scale_path.is_file():
            transform = np.load(scale_path)
        else:
            aria_path = workplace / "aria_from_colmap_transforms.pkl"
            if not aria_path.is_file():
                raise FileNotFoundError(
                    "missing both scale.npy and aria_from_colmap_transforms.pkl"
                )
            with aria_path.open("rb") as stream:
                transforms = pickle.load(stream)
            if "aria01" not in transforms:
                raise KeyError(f"aria01 transform missing from {aria_path}")
            transform = transforms["aria01"]
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"invalid world-from-COLMAP transform {transform.shape}")
        return transform

    def _build_sequence_record(
        self, sequence: Path, max_frames: Optional[int]
    ) -> _SequenceRecord:
        workplace = sequence / "colmap" / "workplace"
        camera_table = self._read_colmap_cameras(workplace / "cameras.txt")
        exo_poses = self._read_colmap_exo_poses(workplace / "images.txt")
        world_from_colmap = self._world_from_colmap(workplace)
        colmap_from_world = np.linalg.inv(world_from_colmap)

        calibrations = []
        for camera_name in sorted(exo_poses):
            image_dir = sequence / "exo" / camera_name / "images"
            if not image_dir.is_dir():
                continue
            _, qvec, tvec, camera_id = exo_poses[camera_name]
            if camera_id not in camera_table:
                continue
            model, width, height, params = camera_table[camera_id]
            if model not in {"OPENCV_FISHEYE", "PINHOLE", "SIMPLE_PINHOLE"}:
                logging.warning(
                    "Harmony4D: unsupported camera model %s in %s/%s",
                    model,
                    sequence.name,
                    camera_name,
                )
                continue

            colmap_extr = np.eye(4, dtype=np.float64)
            colmap_extr[:3, :3] = self._quaternion_to_rotation(qvec)
            colmap_extr[:3, 3] = tvec
            world_extr = colmap_extr @ colmap_from_world
            world_extr = self._orthonormalize_extrinsics(world_extr)
            calibrations.append(
                _CameraCalibration(
                    name=camera_name,
                    image_dir=image_dir,
                    width=width,
                    height=height,
                    model=model,
                    params=params.astype(np.float64),
                    extrinsics=world_extr,
                )
            )

        smpl_dir = sequence / "processed_data" / "smpl"
        frames = [path.stem for path in sorted(smpl_dir.glob("*.npy"))]
        if max_frames is not None:
            frames = frames[: max(0, int(max_frames))]
        relative_path = sequence.relative_to(self.root)
        relative_name = (
            sequence.name
            if relative_path == Path(".")
            else relative_path.as_posix()
        )
        return _SequenceRecord(relative_name, sequence, frames, calibrations)

    def _annotation_path(self, sequence: Path, frame: str) -> Path:
        if self.body_model_type == "smpl":
            return sequence / "processed_data" / "smpl" / f"{frame}.npy"
        assert self.smplx_annotation_root is not None
        relative_sequence = Path(sequence.parent.name) / sequence.name
        return self.smplx_annotation_root / relative_sequence / f"{frame}.npy"

    def _mask_bundle_paths(
        self, record: _SequenceRecord, camera: _CameraCalibration, frame: str
    ) -> Tuple[Path, Path, Path]:
        assert self.person_mask_root is not None
        camera_dir = self.person_mask_root / record.name / camera.name
        return (
            camera_dir / f"{frame}.mask.png",
            camera_dir / "rectified" / f"{frame}.jpg",
            camera_dir / "rectified" / f"{frame}.camera.npz",
        )

    def _view_available(
        self, record: _SequenceRecord, camera: _CameraCalibration, frame: str
    ) -> bool:
        if not (camera.image_dir / f"{frame}.jpg").is_file():
            return False
        if not self.emit_person_mask:
            return True
        return all(
            path.is_file()
            for path in self._mask_bundle_paths(record, camera, frame)
        )

    @staticmethod
    def _intrinsic_and_distortion(
        calibration: _CameraCalibration,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        p = calibration.params
        if calibration.model == "OPENCV_FISHEYE":
            if p.size != 8:
                raise ValueError(f"invalid OPENCV_FISHEYE params: {p.shape}")
            fx, fy, cx, cy = p[:4]
            distortion = p[4:8].reshape(4, 1)
        elif calibration.model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
            distortion = None
        else:  # SIMPLE_PINHOLE: f, cx, cy
            fx, cx, cy = p[:3]
            fy = fx
            distortion = None
        intrinsic = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return intrinsic, distortion

    def _resize_before_undistort(
        self, image: np.ndarray, intrinsic: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.undistort_max_side is None:
            return image, intrinsic
        height, width = image.shape[:2]
        if max(height, width) <= self.undistort_max_side:
            return image, intrinsic
        scale = float(self.undistort_max_side) / float(max(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        sx, sy = new_width / width, new_height / height
        scaled_intrinsic = intrinsic.copy()
        scaled_intrinsic[0, :] *= sx
        scaled_intrinsic[1, :] *= sy
        return resized, scaled_intrinsic

    def _undistort(
        self,
        sequence_name: str,
        calibration: _CameraCalibration,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        intrinsic, distortion = self._intrinsic_and_distortion(calibration)
        image, intrinsic = self._resize_before_undistort(image, intrinsic)
        if distortion is None:
            return image, intrinsic.astype(np.float32)

        height, width = image.shape[:2]
        cache_key = (sequence_name, calibration.name, width, height)
        cached = self._undistort_cache.get(cache_key)
        if cached is None:
            new_intrinsic = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                intrinsic,
                distortion,
                (width, height),
                np.eye(3),
                balance=self.undistort_balance,
                new_size=(width, height),
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                intrinsic,
                distortion,
                np.eye(3),
                new_intrinsic,
                (width, height),
                cv2.CV_16SC2,
            )
            cached = (
                map1,
                map2,
                np.asarray(new_intrinsic, dtype=np.float32),
            )
            if self.undistort_cache_size > 0:
                self._undistort_cache[cache_key] = cached
                self._undistort_cache.move_to_end(cache_key)
                while len(self._undistort_cache) > self.undistort_cache_size:
                    self._undistort_cache.popitem(last=False)
        else:
            self._undistort_cache.move_to_end(cache_key)

        map1, map2, new_intrinsic = cached
        undistorted = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return undistorted, new_intrinsic.copy()

    @staticmethod
    def _project_points(
        points_world: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        rotation = np.asarray(extrinsics[:, :3], dtype=np.float64)
        translation = np.asarray(extrinsics[:, 3], dtype=np.float64)
        camera = points @ rotation.T + translation
        valid_depth = camera[:, 2] > 1e-6
        z = np.where(valid_depth, camera[:, 2], 1.0)
        xy = camera[:, :2] / z[:, None]
        pixels = np.empty_like(xy)
        pixels[:, 0] = intrinsics[0, 0] * xy[:, 0] + intrinsics[0, 2]
        pixels[:, 1] = intrinsics[1, 1] * xy[:, 1] + intrinsics[1, 2]
        return pixels.astype(np.float32), valid_depth

    def _view_order(
        self,
        record: _SequenceRecord,
        frame: str,
        img_per_seq: int,
        ids: Optional[List[int]],
    ) -> List[int]:
        available = [
            i
            for i, camera in enumerate(record.cameras)
            if self._view_available(record, camera, frame)
        ]
        if len(available) < img_per_seq:
            raise RuntimeError(
                f"{record.name}/{frame} has {len(available)} views, needs {img_per_seq}"
            )
        if ids is not None:
            requested = [int(i) for i in ids if int(i) in available]
            remaining = [i for i in available if i not in set(requested)]
            order = requested + remaining
        elif self.fixed_view_sampling:
            order = available
        else:
            order = list(np.random.permutation(available))

        if self.fixed_anchor_camera:
            anchor = next(
                (
                    i
                    for i in available
                    if record.cameras[i].name == self.fixed_anchor_camera
                ),
                None,
            )
            if anchor is not None:
                order = [anchor] + [i for i in order if i != anchor]
        return order[:img_per_seq]

    def get_temporal_length(self, index: int) -> int:
        return self.temporal_clip_length if self.use_temporal_training else 1

    def get_data(
        self,
        seq_index=None,
        seq_name=None,
        ids=None,
        img_per_seq=None,
        aspect_ratio=1.0,
        _resample_depth: int = 0,
    ):
        if not self.use_temporal_training:
            return self._get_single_frame_data(
                seq_index=seq_index,
                seq_name=seq_name,
                ids=ids,
                img_per_seq=img_per_seq,
                aspect_ratio=aspect_ratio,
                _resample_depth=_resample_depth,
            )
        if seq_name is not None:
            raise ValueError("seq_name lookup is not supported for temporal Harmony4D clips")

        clip_index = int(seq_index)
        record_index, frames, frame_ids = self.temporal_samples[clip_index]
        record = self.sequences[record_index]
        requested = int(img_per_seq or len(record.cameras))
        common_views = [
            camera_index
            for camera_index, camera in enumerate(record.cameras)
            if all(self._view_available(record, camera, frame) for frame in frames)
        ]
        if len(common_views) < requested:
            if _resample_depth < 20 and len(self.temporal_samples) > 1:
                return self.get_data(
                    seq_index=random.randrange(len(self.temporal_samples)),
                    img_per_seq=requested,
                    aspect_ratio=aspect_ratio,
                    _resample_depth=_resample_depth + 1,
                )
            raise RuntimeError(
                f"{record.name}/{frames[0]}..{frames[-1]} has "
                f"{len(common_views)} shared views, needs {requested}"
            )

        if ids is not None:
            requested_order = [int(i) for i in ids if int(i) in common_views]
            view_ids = requested_order + [
                i for i in common_views if i not in set(requested_order)
            ]
        elif self.fixed_view_sampling:
            view_ids = list(common_views)
        else:
            view_ids = list(np.random.permutation(common_views))
        if self.fixed_anchor_camera:
            anchor = next(
                (
                    i
                    for i in common_views
                    if record.cameras[i].name == self.fixed_anchor_camera
                ),
                None,
            )
            if anchor is not None:
                view_ids = [anchor] + [i for i in view_ids if i != anchor]
        view_ids = view_ids[:requested]

        frame_batches = [
            self._get_single_frame_data(
                seq_name=f"{record.name}/{frame}",
                ids=view_ids,
                img_per_seq=requested,
                aspect_ratio=aspect_ratio,
                # A temporal clip must never silently substitute an unrelated
                # frame when one selected view fails to load.
                _resample_depth=20,
            )
            for frame in frames
        ]
        return self._combine_temporal_frame_batches(
            record=record,
            frames=frames,
            frame_ids=frame_ids,
            view_ids=view_ids,
            frame_batches=frame_batches,
        )

    def _combine_temporal_frame_batches(
        self, *, record, frames, frame_ids, view_ids, frame_batches
    ):
        T = len(frame_batches)
        V = len(view_ids)
        combined = {
            "seq_name": (
                f"harmony4d_temporal_{record.name.replace('/', '_')}_"
                f"{frames[0]}_{frames[-1]}"
            ),
            "frame_num": T * V,
            "temporal_num_frames": T,
            "views_per_frame": V,
            "frame_ids": np.asarray(frame_ids, dtype=np.int64),
            "view_ids": np.asarray(view_ids, dtype=np.int64),
            "ids": np.tile(np.asarray(view_ids, dtype=np.int64), T),
            "person_keys": list(frame_batches[0]["person_keys"]),
            "num_people": np.asarray(
                [int(np.asarray(batch["num_people"]).item()) for batch in frame_batches],
                dtype=np.int64,
            ),
            "smpl_model_type": self.body_model_type,
        }

        view_list_keys = (
            "images", "depths", "extrinsics", "intrinsics", "cam_points",
            "world_points", "point_masks", "original_sizes", "image_filenames",
        )
        for key in view_list_keys:
            if all(key in batch for batch in frame_batches):
                combined[key] = [
                    value for batch in frame_batches for value in batch[key]
                ]

        view_array_keys = (
            "smpl_joints2d", "smpl_joints3d_world",
            "smpl_joints2d_confidence", "person_mask", "cam_ids",
        )
        for key in view_array_keys:
            if all(key in batch for batch in frame_batches):
                combined[key] = np.concatenate(
                    [np.asarray(batch[key]) for batch in frame_batches], axis=0
                )

        person_keys = (
            "smpl_pose", "smpl_beta", "smpl_trans", "smpl_gender", "has_smpl",
        )
        for key in person_keys:
            if all(key in batch for batch in frame_batches):
                combined[key] = np.stack(
                    [np.asarray(batch[key]) for batch in frame_batches], axis=0
                )
        return combined

    def _get_single_frame_data(
        self,
        seq_index=None,
        seq_name=None,
        ids=None,
        img_per_seq=None,
        aspect_ratio=1.0,
        _resample_depth: int = 0,
    ):
        if seq_name is not None:
            matches = [i for i, (s, f) in enumerate(self.samples) if f"{self.sequences[s].name}/{f}" == seq_name]
            if not matches:
                raise KeyError(seq_name)
            sample_index = matches[0]
        else:
            sample_index = int(seq_index)

        record_index, frame = self.samples[sample_index]
        record = self.sequences[record_index]
        n_views = int(img_per_seq or len(record.cameras))
        if n_views < 1:
            raise ValueError("img_per_seq must be positive")

        try:
            view_ids = self._view_order(record, frame, n_views, ids)
        except RuntimeError:
            if _resample_depth < 20 and len(self.samples) > 1:
                return self._get_single_frame_data(
                    seq_index=random.randrange(len(self.samples)),
                    img_per_seq=n_views,
                    aspect_ratio=aspect_ratio,
                    _resample_depth=_resample_depth + 1,
                )
            raise

        smpl_path = self._annotation_path(record.path, frame)
        smpl = np.load(smpl_path, allow_pickle=True).item()
        people_keys = [key for key in self.PEOPLE if key in smpl]
        if len(people_keys) != len(self.PEOPLE):
            raise KeyError(
                f"{smpl_path} expected people {self.PEOPLE}, got {tuple(smpl)}"
            )

        padded_people = self.max_num_people
        person_count = len(people_keys)
        poses = np.zeros((padded_people, 72), dtype=np.float32)
        betas = np.zeros((padded_people, 10), dtype=np.float32)
        translations = np.zeros((padded_people, 3), dtype=np.float32)
        joints_world = np.zeros((padded_people, 24, 3), dtype=np.float32)
        has_smpl = np.zeros((padded_people,), dtype=np.float32)
        genders = np.full((padded_people,), 2, dtype=np.int64)  # neutral SMPL
        for person_idx, key in enumerate(people_keys):
            person = smpl[key]
            global_orient = np.asarray(
                person["global_orient"], dtype=np.float32
            ).reshape(3)
            body_pose = np.asarray(
                person["body_pose"], dtype=np.float32
            ).reshape(-1)
            if self.body_model_type == "smplx":
                if body_pose.size not in {63, 69}:
                    raise ValueError(
                        "SMPL-X body_pose must have 63 or 69 values, "
                        f"got {body_pose.size}"
                    )
                body_pose69 = np.zeros(69, dtype=np.float32)
                body_pose69[:63] = body_pose[:63]
            else:
                if body_pose.size != 69:
                    raise ValueError(
                        f"SMPL body_pose must have 69 values, got {body_pose.size}"
                    )
                body_pose69 = body_pose
            poses[person_idx] = np.concatenate([global_orient, body_pose69])
            betas[person_idx] = np.asarray(
                person["betas"], dtype=np.float32
            ).reshape(-1)[:10]
            translations[person_idx] = np.asarray(
                person["transl"], dtype=np.float32
            ).reshape(3)
            joints_world[person_idx] = np.asarray(
                person["joints"], dtype=np.float32
            )[:24]
            has_smpl[person_idx] = 1.0

        target_shape = self.get_target_shape(aspect_ratio)
        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []
        joints2d_views, confidence_views = [], []
        image_filenames, original_sizes, camera_ids = [], [], []
        person_mask_views = []

        for view_id in view_ids:
            camera = record.cameras[view_id]
            instance_mask = None
            camera_metadata_path = None
            if self.emit_person_mask:
                mask_path, image_path, camera_metadata_path = self._mask_bundle_paths(
                    record, camera, frame
                )
            else:
                image_path = camera.image_dir / f"{frame}.jpg"
            image = read_image_cv2(str(image_path))
            if image is None:
                if _resample_depth < 20 and len(self.samples) > 1:
                    return self._get_single_frame_data(
                        seq_index=random.randrange(len(self.samples)),
                        img_per_seq=n_views,
                        aspect_ratio=aspect_ratio,
                        _resample_depth=_resample_depth + 1,
                    )
                raise RuntimeError(f"Could not read {image_path}")

            if self.emit_person_mask:
                instance_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if instance_mask is None:
                    raise RuntimeError(f"Could not read {mask_path}")
                if instance_mask.shape != image.shape[:2]:
                    raise ValueError(
                        f"Harmony4D RGB/mask shape mismatch for {record.name}/"
                        f"{camera.name}/{frame}: {image.shape[:2]} vs "
                        f"{instance_mask.shape}"
                    )
                invalid_labels = np.setdiff1d(
                    np.unique(instance_mask), np.asarray([0, 1, 2], dtype=np.uint8)
                )
                if invalid_labels.size:
                    raise ValueError(
                        f"Invalid instance labels in {mask_path}: "
                        f"{invalid_labels.tolist()}"
                    )
                assert camera_metadata_path is not None
                with np.load(camera_metadata_path, allow_pickle=False) as metadata:
                    intrinsic = np.asarray(metadata["intrinsic"], dtype=np.float32)
                    stored_extrinsic = np.asarray(
                        metadata["extrinsics"], dtype=np.float32
                    )
                    rectified_shape = tuple(
                        np.asarray(metadata["rectified_shape"], dtype=np.int32).tolist()
                    )
                if rectified_shape != image.shape[:2]:
                    raise ValueError(
                        f"Rectified metadata/image mismatch in {camera_metadata_path}: "
                        f"{rectified_shape} vs {image.shape[:2]}"
                    )
                if stored_extrinsic.shape != (3, 4) or not np.allclose(
                    stored_extrinsic, camera.extrinsics, atol=1e-5
                ):
                    raise ValueError(
                        f"Camera extrinsics mismatch in {camera_metadata_path}"
                    )
                extrinsic = stored_extrinsic.copy()
            else:
                image, intrinsic = self._undistort(record.name, camera, image)
                extrinsic = camera.extrinsics.copy()
            original_size = np.asarray(image.shape[:2], dtype=np.int64)
            depth_map = np.zeros(image.shape[:2], dtype=np.float32)
            extra_maps = (
                {"person_mask": instance_mask.astype(np.float32)}
                if instance_mask is not None
                else None
            )
            (
                image,
                depth_map,
                extrinsic,
                intrinsic,
                world_coords,
                camera_coords,
                point_mask,
                _unused_track,
                _unused_in_frame,
            ) = self.process_one_image(
                image,
                depth_map,
                extrinsic,
                intrinsic,
                original_size,
                target_shape,
                track=None,
                filepath=str(image_path),
                extra_maps=extra_maps,
            )

            # The camera pose encoding stores FoV but not principal point and
            # reconstructs K with (cx, cy)=(W/2, H/2). BaseDataset's PIL
            # pixel-center bookkeeping leaves the principal point ~0.5--1 px
            # away from that center. Apply the equivalent sub-pixel image shift
            # so the RGB, K, and joints agree with the camera-head convention.
            height_final, width_final = image.shape[:2]
            shift_x = width_final / 2.0 - float(intrinsic[0, 2])
            shift_y = height_final / 2.0 - float(intrinsic[1, 2])
            if abs(shift_x) > 1e-6 or abs(shift_y) > 1e-6:
                affine = np.asarray(
                    [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]],
                    dtype=np.float32,
                )
                image = cv2.warpAffine(
                    image,
                    affine,
                    (width_final, height_final),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                if extra_maps is not None:
                    extra_maps["person_mask"] = cv2.warpAffine(
                        extra_maps["person_mask"],
                        affine,
                        (width_final, height_final),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                intrinsic = intrinsic.copy()
                intrinsic[0, 2] = width_final / 2.0
                intrinsic[1, 2] = height_final / 2.0

            # Reproject with the FINAL K/E instead of retaining BaseDataset's
            # transformed track. Its PIL pixel-center correction updates K by
            # ~0.2 px but scales track coordinates without the same correction;
            # regenerating from 3-D makes loader GT and the loss projection use
            # exactly one camera convention.
            projected, final_valid_depth = self._project_points(
                joints_world.reshape(-1, 3), extrinsic, intrinsic
            )
            projected = projected.reshape(padded_people, 24, 2).astype(np.float32)
            confidence = (
                final_valid_depth.reshape(padded_people, 24)
                & np.isfinite(projected).all(axis=-1)
                & (projected[..., 0] >= 0)
                & (projected[..., 0] < width_final)
                & (projected[..., 1] >= 0)
                & (projected[..., 1] < height_final)
            ).astype(np.float32)
            confidence *= has_smpl[:, None]

            if extra_maps is not None:
                mask_stride = self.person_mask_stride or self.patch_size
                mask_h = height_final // mask_stride
                mask_w = width_final // mask_stride
                per_person_mask = np.zeros(
                    (padded_people, mask_h, mask_w), dtype=np.float32
                )
                for person_idx in range(person_count):
                    per_person_mask[person_idx] = rasterize_person_patch_mask(
                        extra_maps["person_mask"], person_idx + 1, mask_h, mask_w
                    )
                person_mask_views.append(per_person_mask)

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extrinsic.astype(np.float32))
            intrinsics.append(intrinsic.astype(np.float32))
            cam_points.append(camera_coords)
            world_points.append(world_coords)
            point_masks.append(point_mask)
            joints2d_views.append(projected)
            confidence_views.append(confidence)
            image_filenames.append(str(image_path))
            original_sizes.append(original_size)
            camera_ids.append(int(camera.name[3:]) if camera.name[3:].isdigit() else view_id)

        repeated_joints_world = np.repeat(
            joints_world[None, ...], len(view_ids), axis=0
        ).astype(np.float32)
        result = {
            "seq_name": f"harmony4d_{record.name.replace('/', '_')}_{frame}",
            "ids": np.asarray(view_ids, dtype=np.int64),
            "frame_num": len(view_ids),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "smpl_pose": poses,
            "smpl_beta": betas,
            "smpl_trans": translations,
            "smpl_joints2d": np.stack(joints2d_views).astype(np.float32),
            "smpl_joints3d_world": repeated_joints_world,
            "smpl_joints2d_confidence": np.stack(confidence_views).astype(np.float32),
            "smpl_gender": genders,
            "has_smpl": has_smpl,
            "num_people": np.asarray(person_count, dtype=np.int64),
            "person_keys": people_keys,
            "image_filenames": image_filenames,
            "cam_ids": np.asarray(camera_ids, dtype=np.int64),
            "smpl_model_type": self.body_model_type,
        }
        if person_mask_views:
            result["person_mask"] = np.stack(person_mask_views).astype(np.float32)
        return result
