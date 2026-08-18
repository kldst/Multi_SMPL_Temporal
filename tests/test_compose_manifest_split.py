from pathlib import Path

from training.data.datasets.sys_smpl_multi import SysSMPLMultiDataset


def make_dataset(split: str, fraction: float, seed: int = 42):
    dataset = SysSMPLMultiDataset.__new__(SysSMPLMultiDataset)
    dataset.split = split
    dataset.val_sequence_fraction = fraction
    dataset.split_seed = seed
    return dataset


def test_compose_sequence_split_is_disjoint_and_complete():
    paths = [Path(f"sequence_{index:03d}/manifest.pkl") for index in range(20)]
    train = make_dataset("train", 0.2)._split_compose_manifests(paths)
    val = make_dataset("val", 0.2)._split_compose_manifests(paths)

    assert len(train) == 16
    assert len(val) == 4
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(paths)
    assert val == make_dataset("val", 0.2)._split_compose_manifests(paths)


def test_zero_fraction_retains_all_training_sequences():
    paths = [Path("a/manifest.pkl"), Path("b/manifest.pkl")]

    assert make_dataset("train", 0.0)._split_compose_manifests(paths) == paths
    assert make_dataset("val", 0.0)._split_compose_manifests(paths) == []
    assert make_dataset("test", 0.5)._split_compose_manifests(paths) == paths
