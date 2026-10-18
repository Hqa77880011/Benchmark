"""Regression checks for synthesis, data pairing and evaluation boundaries."""

import importlib.util
import subprocess
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synthesis_pipeline"))

from compute import calculate_metrics
from lithodefect.synthesis import Synthesizer
from lithodefect.split import split_dataset


def make_seeds(root):
    """Write a tiny fixture with a visible defect and two background sources."""
    image = np.full((20, 20, 3), 180, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:16, 4:16] = 255
    for folder, name, array in [
        ("defects/images/arching", "seed.png", image),
        ("defects/masks/arching", "seed_mask.png", mask),
        ("backgrounds", "chip_001.png", np.full((64, 64, 3), 60, np.uint8)),
        ("backgrounds", "shen_001.png", np.full((64, 64, 3), 90, np.uint8)),
    ]:
        dest = root / folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(dest), array)


def test_pipeline_from_another_working_directory(tmp_path):
    """The shipped config must run all steps independently of the shell cwd."""
    make_seeds(tmp_path / "data")
    config = yaml.safe_load((ROOT / "synthesis_pipeline/config.yaml").read_text(encoding="utf-8"))
    config["category_mapping"] = {"arching": ["ArchDeform"]}
    config["class_mapping"] = {"ArchDeform": 0}
    for group in config["bg_groups"].values():
        group["num_aug"] = 10
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "synthesis_pipeline/run_pipeline.py"),
         "--config", str(config_path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "output"
    images = list((output / "synthetic_defects").glob("*/img/*/*.png"))
    assert len(images) == 20
    split_images = list((output / "synthetic_defects_split").rglob("img/*/*.png"))
    assert len(split_images) == 20
    assert len({p.name for p in split_images}) == 20
    yolo_config = yaml.safe_load((output / "synthetic_yolo/data.yaml").read_text(encoding="utf-8"))
    assert Path(yolo_config["path"]).is_absolute()
    for path in (output / "swin_unet_data").glob("*/mask/*.png"):
        assert set(np.unique(cv2.imread(str(path), 0))) <= {0, 1}


def test_synthesis_reproducible_and_binary(tmp_path):
    make_seeds(tmp_path / "data")
    config = {
        "defect_img_dir": str(tmp_path / "data/defects/images"),
        "defect_mask_dir": str(tmp_path / "data/defects/masks"),
        "bg_dir": str(tmp_path / "data/backgrounds"),
        "output_root": str(tmp_path / "first"),
        "category_mapping": {"arching": ["ArchDeform"]},
        "bg_groups": {"private": {"prefixes": ["chip_"], "num_aug": 3}},
        "synthesis_params": {"enable_feather": True, "enable_distortion": True,
                             "distortion_types": ["barrel"]},
        "seed": 42,
    }
    Synthesizer(config).run()
    config["output_root"] = str(tmp_path / "second")
    Synthesizer(config).run()
    for image in (tmp_path / "first").rglob("*.png"):
        assert image.read_bytes() == (tmp_path / "second" / image.relative_to(tmp_path / "first")).read_bytes()
        if "mask" in image.parts:
            assert set(np.unique(cv2.imread(str(image), 0))) <= {0, 255}


def test_missing_prediction_fails_before_reporting(tmp_path):
    (tmp_path / "gt").mkdir()
    (tmp_path / "pred").mkdir()
    cv2.imwrite(str(tmp_path / "gt/missing.png"), np.ones((8, 8), np.uint8))
    result = subprocess.run(
        [sys.executable, str(ROOT / "compute.py"), "--gt-dir", str(tmp_path / "gt"),
         "--pred-dir", str(tmp_path / "pred"), "--output-dir", str(tmp_path / "metrics")],
        capture_output=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "metrics/metrics_results.txt").exists()


def test_metrics_hand_calculated():
    pred = np.array([[1, 1], [0, 0]])
    true = np.array([[1, 0], [1, 0]])
    values = calculate_metrics(pred, true)
    assert values == pytest.approx({"IoU": 1 / 3, "Dice": .5,
                                   "Precision": .5, "Recall": .5})


def test_split_rejects_empty_input(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        split_dataset(tmp_path / "missing", tmp_path / "out")


def test_split_rejects_impossible_background_ratio(tmp_path):
    for i in range(4):
        for kind in ["img", "mask"]:
            dest = tmp_path / "data/private" / kind / "ArchDeform" / f"{i}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), np.ones((8, 8), np.uint8))
    with pytest.raises(ValueError, match="public"):
        split_dataset(tmp_path / "data", tmp_path / "out")


def load_unet_dataset():
    spec = importlib.util.spec_from_file_location(
        "litho_swin_dataset", ROOT / "unet/datasets/dataset_synapse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_transform_is_deterministic():
    module = load_unet_dataset()
    transform = module.RandomGenerator([16, 16], training=False)
    sample = {"image": np.arange(16 * 16 * 3).reshape(16, 16, 3),
              "label": np.eye(16, dtype=np.uint8)}
    a, b = transform(sample), transform(sample)
    assert np.array_equal(a["image"], b["image"])
    assert np.array_equal(a["label"], b["label"])


def test_swin_loader_reads_exported_png_without_resplitting(tmp_path):
    module = load_unet_dataset()
    for split in ["train", "val"]:
        for kind in ["img", "mask"]:
            dest = tmp_path / split / kind / f"{split}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            arr = np.ones((16, 16, 3) if kind == "img" else (16, 16), np.uint8)
            cv2.imwrite(str(dest), arr)
    train = module.Synapse_dataset(tmp_path, None, "train")
    val = module.Synapse_dataset(tmp_path, None, "val")
    assert len(train) == len(val) == 1
    assert train[0]["case_name"] == "train"
    assert val[0]["case_name"] == "val"


def test_training_and_validation_use_different_samples(tmp_path):
    sys.path.insert(0, str(ROOT / "unet"))
    from trainer import build_loaders

    for split, value in [("train", 40), ("val", 200)]:
        for kind in ["img", "mask"]:
            dest = tmp_path / split / kind / f"{split}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            array = np.full((16, 16, 3), value, np.uint8) if kind == "img" else np.zeros((16, 16), np.uint8)
            cv2.imwrite(str(dest), array)
    args = SimpleNamespace(root_path=tmp_path, list_dir=None, img_size=16,
                           batch_size=1, n_gpu=1, num_workers=0)
    train, val = build_loaders(args)
    assert train.dataset[0]["case_name"] == "train"
    assert val.dataset[0]["case_name"] == "val"
    assert float(next(iter(val))["image"].mean()) == 200


def test_evaluation_does_not_resize_silently():
    with pytest.raises(ValueError, match="shapes"):
        calculate_metrics(np.ones((2, 2)), np.ones((4, 4)))


def test_empty_mask_metric_does_not_reward_false_positives():
    spec = importlib.util.spec_from_file_location("swin_utils", ROOT / "unet/utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.calculate_metric_percase(np.ones((2, 2)), np.zeros((2, 2)))[0] == 0


def test_dataset_keeps_all_pairs_in_existing_split(tmp_path):
    from lithodefect.dataset import LithoDefectDataset

    for i in range(3):
        for kind in ["img", "mask"]:
            path = tmp_path / kind / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((8, 8), 255, np.uint8))
    dataset = LithoDefectDataset(tmp_path)
    assert len(dataset) == 3
    image, mask = dataset[0]
    assert tuple(image.shape) == (3, 8, 8)
    assert set(mask.unique().tolist()) == {1}


def test_cpu_training_singleton_batch_and_validation(tmp_path):
    import torch
    sys.path.insert(0, str(ROOT / "unet"))
    from trainer import trainer_synapse

    for split, count in [("train", 20), ("val", 2)]:
        for i in range(count):
            for kind in ["img", "mask"]:
                path = tmp_path / "data" / split / kind / f"{i}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                array = np.full((8, 8, 3), 1, np.uint8) if kind == "img" else np.eye(8, dtype=np.uint8)
                cv2.imwrite(str(path), array)
    args = SimpleNamespace(root_path=tmp_path / "data", list_dir=None,
                           img_size=8, batch_size=1, n_gpu=1, num_workers=0,
                           base_lr=.01, num_classes=2, max_epochs=1, eval_interval=1)
    output = tmp_path / "model"
    output.mkdir()
    model = torch.nn.Conv2d(3, 2, kernel_size=1)
    assert trainer_synapse(args, model, str(output)) == "Training Finished!"
    assert (output / "best_model.pth").is_file()
