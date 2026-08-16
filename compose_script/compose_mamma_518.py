#!/usr/bin/env python
"""Build a geometry-correct 518x518 cache for raw multi-view MAMMA data.

The converter intentionally uses the same principal-point crop, resize, mask
transform, and intrinsic update as ``BaseDataset.process_one_image``.  It never
modifies the source tree.  Each output sequence contains lossless RGB/mask PNGs,
per-view camera NPZ files, and a compact manifest used by
``SysSMPLMultiDataset`` without scanning the large ``*.data.pyd`` files again.

The command is resumable: completed image/mask/camera bundles are reused, while
the compact manifest is rebuilt/merged from source annotations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO / "training"
for _path in (str(REPO), str(TRAINING_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(REPO)

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from training.data.base_dataset import BaseDataset  # noqa: E402
from training.data.dataset_util import read_image_cv2  # noqa: E402
from training.data.datasets.sys_smpl_multi import SysSMPLMultiDataset  # noqa: E402


FORMAT_NAME = "mamma_compose_518"
FORMAT_VERSION = 1
DEFAULT_SOURCE = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/mamma/mamma"
)
DEFAULT_OUTPUT = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/mamma/mamma_compose"
)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_png(path: Path, image: np.ndarray, compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.png")
    ok = cv2.imwrite(
        str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)]
    )
    if not ok:
        raise RuntimeError(f"Failed to write {temporary}")
    os.replace(temporary, path)


def atomic_camera_npz(
    path: Path,
    *,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_size: np.ndarray,
    track_offset: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
        original_size=np.asarray(original_size, dtype=np.int64),
        track_offset=np.asarray(track_offset, dtype=np.float32),
    )
    os.replace(temporary, path)


def find_mask(data_path: Path) -> Optional[Path]:
    frame = data_path.name[: -len(".data.pyd")]
    for suffix in (".mask.jpg", ".mask.png", ".mask.jpeg"):
        candidate = data_path.with_name(frame + suffix)
        if candidate.is_file():
            return candidate
    return None


def numeric_sort_key(name: str):
    import re

    matches = re.findall(r"\d+", str(name))
    return (int(matches[-1]) if matches else -1, str(name))


def normalize_gender_for_manifest(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else "neutral"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return value


def make_preprocessor(image_size: int, patch_size: int) -> BaseDataset:
    common = OmegaConf.create(
        {
            "img_size": int(image_size),
            "patch_size": int(patch_size),
            "rescale": True,
            "rescale_aug": False,
            "landscape_check": False,
            "emit_dense_geometry": False,
            "profile_data_loading": False,
            "augs": {"scales": None},
        }
    )
    preprocessor = BaseDataset(common)
    # BaseDataset deliberately initializes only geometric fields; concrete
    # datasets normally copy this flag themselves. The offline conversion must
    # be deterministic and match validation geometry (no random scale crop).
    preprocessor.training = False
    return preprocessor


def bundle_is_complete(
    rgb_path: Path, mask_path: Optional[Path], camera_path: Path
) -> bool:
    if not rgb_path.is_file() or not camera_path.is_file():
        return False
    if mask_path is not None and not mask_path.is_file():
        return False
    try:
        camera = np.load(camera_path)
        return (
            camera["intrinsics"].shape == (3, 3)
            and camera["extrinsics"].shape == (3, 4)
            and camera["original_size"].size == 2
            and camera["track_offset"].shape == (2,)
        )
    except Exception:
        return False


def compose_view(
    *,
    preprocessor: BaseDataset,
    image_path: Path,
    source_mask_path: Optional[Path],
    output_rgb: Path,
    output_mask: Optional[Path],
    output_camera: Path,
    camera: dict[str, np.ndarray],
    target_shape: np.ndarray,
    png_compression: int,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not overwrite and bundle_is_complete(output_rgb, output_mask, output_camera):
        cached = np.load(output_camera)
        return (
            cached["intrinsics"].astype(np.float32),
            cached["extrinsics"].astype(np.float32),
            cached["original_size"].astype(np.int64),
            cached["track_offset"].astype(np.float32),
        )

    image = read_image_cv2(str(image_path))
    if image is None:
        raise RuntimeError(f"Unreadable RGB: {image_path}")
    original_size = np.asarray(image.shape[:2], dtype=np.int64)
    extra_maps = {}
    if source_mask_path is not None:
        mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unreadable mask: {source_mask_path}")
        extra_maps["person_mask"] = mask

    (
        processed,
        _,
        extrinsics,
        intrinsics,
        _,
        _,
        _,
        _,
        _,
    ) = preprocessor.process_one_image(
        image=image,
        depth_map=None,
        extri_opencv=camera["extrinsics"],
        intri_opencv=camera["intrinsics"],
        original_size=original_size,
        target_image_shape=target_shape,
        track=None,
        filepath=str(image_path),
        extra_maps=extra_maps or None,
    )
    if tuple(processed.shape[:2]) != tuple(int(x) for x in target_shape):
        raise ValueError(
            f"Processed shape {processed.shape[:2]} != {tuple(target_shape)}"
        )

    atomic_png(
        output_rgb,
        cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
        png_compression,
    )
    if output_mask is not None:
        processed_mask = extra_maps.get("person_mask")
        if processed_mask is None:
            raise RuntimeError(f"Mask transform was lost for {source_mask_path}")
        atomic_png(
            output_mask,
            np.rint(processed_mask).astype(np.uint8),
            png_compression,
        )
    # ``resize_image_depth_and_intrinsic`` uses OpenCV's half-pixel convention
    # for K, while its legacy track path scales coordinates without that 0.5
    # shift. Preserve the resulting sub-pixel GT convention so cached batches
    # reproduce the raw loader and its loss values exactly.
    scale_x = float(intrinsics[0, 0] / camera["intrinsics"][0, 0])
    scale_y = float(intrinsics[1, 1] / camera["intrinsics"][1, 1])
    track_offset = np.asarray(
        [0.5 * (1.0 - scale_x), 0.5 * (1.0 - scale_y)],
        dtype=np.float32,
    )
    atomic_camera_npz(
        output_camera,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        original_size=original_size,
        track_offset=track_offset,
    )
    return (
        np.asarray(intrinsics, dtype=np.float32),
        np.asarray(extrinsics, dtype=np.float32),
        original_size,
        track_offset,
    )


def load_existing_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
        if value.get("format") == FORMAT_NAME and int(value.get("version", -1)) == FORMAT_VERSION:
            return value
    except Exception:
        pass
    return {}


def compose_sequence(
    *,
    source_root: Path,
    output_root: Path,
    sequence: Path,
    preprocessor: BaseDataset,
    image_size: int,
    patch_size: int,
    max_frames: Optional[int],
    png_compression: int,
    overwrite: bool,
) -> dict[str, Any]:
    relative_sequence = sequence.relative_to(source_root)
    output_sequence = output_root / relative_sequence
    manifest_path = output_sequence / "manifest.pkl"
    existing = load_existing_manifest(manifest_path)
    manifest_frames = dict(existing.get("frames", {}))

    grouped = SysSMPLMultiDataset._group_raw_mamma_frames(sequence)
    frame_items = sorted(grouped.items(), key=lambda item: numeric_sort_key(item[0]))
    if max_frames is not None:
        frame_items = frame_items[: max(0, int(max_frames))]
    target_shape = preprocessor.get_target_shape(1.0)
    counters = {
        "candidate_frames": len(frame_items),
        "kept_frames": 0,
        "views": 0,
        "failed_views": 0,
    }

    for frame_index, (frame, view_names) in enumerate(frame_items, 1):
        loaded_views = []
        for view_name in view_names:
            data_path = sequence / view_name / f"{frame}.data.pyd"
            image_path = SysSMPLMultiDataset._raw_image_path(sequence, view_name, frame)
            if image_path is None or not data_path.is_file():
                counters["failed_views"] += 1
                continue
            try:
                with data_path.open("rb") as stream:
                    frame_people = pickle.load(stream)
                if not isinstance(frame_people, dict) or not frame_people:
                    raise ValueError("empty/non-dict pyd")
                first_person = next(iter(frame_people.values()))
                camera_raw = SysSMPLMultiDataset._parse_camera(
                    first_person["cam_int"], first_person["cam_ext"]
                )
                if camera_raw is None:
                    raise ValueError("invalid camera")
                source_mask = find_mask(data_path)
                view_output = output_sequence / view_name
                output_rgb = view_output / f"{frame}.png"
                output_mask = (
                    view_output / f"{frame}.mask.png"
                    if source_mask is not None
                    else None
                )
                output_camera = view_output / f"{frame}.camera.npz"
                intrinsics, extrinsics, original_size, track_offset = compose_view(
                    preprocessor=preprocessor,
                    image_path=image_path,
                    source_mask_path=source_mask,
                    output_rgb=output_rgb,
                    output_mask=output_mask,
                    output_camera=output_camera,
                    camera=camera_raw,
                    target_shape=target_shape,
                    png_compression=png_compression,
                    overwrite=overwrite,
                )
                loaded_views.append(
                    {
                        "view_name": str(view_name),
                        "frame_people": frame_people,
                        "image_path": output_rgb.relative_to(output_sequence).as_posix(),
                        "mask_path": (
                            output_mask.relative_to(output_sequence).as_posix()
                            if output_mask is not None
                            else None
                        ),
                        "camera_path": output_camera.relative_to(output_sequence).as_posix(),
                        "source_data_path": str(data_path),
                        "intrinsics": intrinsics,
                        "extrinsics": extrinsics,
                        "original_size": original_size,
                        "track_offset": track_offset,
                    }
                )
            except Exception as exc:
                counters["failed_views"] += 1
                logging.warning("Skipping %s/%s/%s: %s", relative_sequence, view_name, frame, exc)

        if not loaded_views:
            continue
        person_ids = sorted(
            {person_id for view in loaded_views for person_id in view["frame_people"]},
            key=lambda value: int(value),
        )
        people_params = {}
        for person_id in person_ids:
            person = next(
                view["frame_people"][person_id]
                for view in loaded_views
                if person_id in view["frame_people"]
            )
            people_params[person_id] = {
                "person_key": f"person_{int(person_id):02d}",
                "pyd_key": person_id,
                "smpl_pose": np.asarray(person["pose_world"], dtype=np.float32).reshape(-1)[:72],
                "smpl_beta": np.asarray(person["shape"], dtype=np.float32).reshape(-1)[:10],
                "smpl_trans": np.asarray(person["trans_world"], dtype=np.float32).reshape(-1)[:3],
                "gender": normalize_gender_for_manifest(person.get("gender", "neutral")),
                "person_idx": int(person.get("person_idx", int(person_id))),
            }

        annotations = []
        for view in loaded_views:
            people = [
                dict(
                    people_params[person_id],
                    visible_in_view=person_id in view["frame_people"],
                )
                for person_id in person_ids
            ]
            annotations.append(
                {
                    "view_name": view["view_name"],
                    "image_path": view["image_path"],
                    "mask_path": view["mask_path"],
                    "camera_path": view["camera_path"],
                    "source_data_path": view["source_data_path"],
                    "intrinsics": view["intrinsics"],
                    "extrinsics": view["extrinsics"],
                    "original_size": view["original_size"],
                    "track_offset": view["track_offset"],
                    "people": people,
                    "num_people": len(people),
                    "raw_mamma": True,
                    "composed_mamma": True,
                    "preprocessed_518": True,
                }
            )
        manifest_frames[str(frame)] = annotations
        counters["kept_frames"] += 1
        counters["views"] += len(annotations)
        if frame_index % 25 == 0 or frame_index == len(frame_items):
            logging.info(
                "%s: %d/%d selected frames composed",
                relative_sequence,
                frame_index,
                len(frame_items),
            )

    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source_root": str(source_root),
        "source_sequence": str(sequence),
        "sequence_rel": relative_sequence.as_posix(),
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "target_shape": [int(x) for x in target_shape],
        "frames": manifest_frames,
    }
    atomic_pickle(manifest_path, manifest)
    counters.update(
        {
            "sequence": relative_sequence.as_posix(),
            "manifest": str(manifest_path),
            "manifest_total_frames": len(manifest_frames),
            "source_total_frames": len(grouped),
            "complete_sequence": (
                len(manifest_frames) >= len(grouped)
                and counters["failed_views"] == 0
            ),
        }
    )
    return counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Limit to a top-level dataset name; repeatable.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Limit to a sequence basename or source-relative path; repeatable.",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--png-compression", type=int, default=1, choices=range(0, 10))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output-root must not be inside/equal to source-root")
    output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    all_sequences = SysSMPLMultiDataset._find_raw_mamma_sequences(source_root)
    sequences = list(all_sequences)
    if args.dataset:
        allowed = set(args.dataset)
        sequences = [
            sequence
            for sequence in sequences
            if sequence.relative_to(source_root).parts[0] in allowed
        ]
    if args.sequence:
        allowed = set(args.sequence)
        sequences = [
            sequence
            for sequence in sequences
            if sequence.name in allowed
            or sequence.relative_to(source_root).as_posix() in allowed
        ]
    if args.max_sequences is not None:
        sequences = sequences[: max(0, int(args.max_sequences))]
    if not sequences:
        raise ValueError("No matching raw MAMMA sequences")

    preprocessor = make_preprocessor(args.image_size, args.patch_size)
    started = time.perf_counter()
    results = []
    for index, sequence in enumerate(sequences, 1):
        logging.info("Composing sequence %d/%d: %s", index, len(sequences), sequence)
        sequence_started = time.perf_counter()
        result = compose_sequence(
            source_root=source_root,
            output_root=output_root,
            sequence=sequence,
            preprocessor=preprocessor,
            image_size=args.image_size,
            patch_size=args.patch_size,
            max_frames=args.max_frames,
            png_compression=args.png_compression,
            overwrite=args.overwrite,
        )
        result["seconds"] = time.perf_counter() - sequence_started
        results.append(result)

    is_full_source_run = (
        not args.dataset
        and not args.sequence
        and args.max_sequences is None
        and args.max_frames is None
        and len(sequences) == len(all_sequences)
        and all(result["complete_sequence"] for result in results)
    )
    summary = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "selected_sequences": len(sequences),
        "selected_max_frames": args.max_frames,
        "total_seconds": time.perf_counter() - started,
        "full_source_run": is_full_source_run,
        "sequences": results,
    }
    atomic_json(output_root / "last_compose_run.json", summary)
    if is_full_source_run:
        atomic_json(
            output_root / "compose_complete.json",
            {
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "source_root": str(source_root),
                "sequences": len(results),
                "frames": sum(result["manifest_total_frames"] for result in results),
                "views": sum(result["views"] for result in results),
                "completed_unix_time": time.time(),
            },
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
