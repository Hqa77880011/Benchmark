"""Checks for the public dataset-to-training and prediction-to-metrics paths."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("directory", ["ascii", "数据"])
def test_released_archive_layout_can_be_prepared(tmp_path, directory):
    tmp_path = tmp_path / directory
    for category in ["ArchDeform", "Block"]:
        for i in range(10):
            for kind in ["img", "mask"]:
                path = tmp_path / "data" / kind / category / f"{category}_{i}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                shape = (32, 32, 3) if kind == "img" else (32, 32)
                Image.fromarray(np.full(shape, 255, np.uint8)).save(path)
    output = tmp_path / "prepared"
    result = subprocess.run(
        [sys.executable, str(ROOT / "prepare_dataset.py"), "--source", str(tmp_path / "data"),
         "--output", str(output)], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    sets = [set(manifest["splits"][key]) for key in ["train", "val", "test"]]
    assert [len(items) for items in sets] == [14, 2, 4]
    assert len(set.union(*sets)) == 20
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(list((output / "yolo/images/test").glob("*.png"))) == 4
    import yaml
    assert yaml.safe_load((output / "yolo/data.yaml").read_text(encoding="utf-8"))["test"] == "images/test"
    assert len(list((output / "swin_unet/test/mask").glob("*.png"))) == 4
    assert set(np.unique(np.asarray(Image.open(next((output / "swin_unet/test/mask").glob("*.png")))))) == {1}


def test_swin_checkpoint_path_independent_of_working_directory():
    module = load_module("swin_reproduction_config", "unet/config.py")
    args = SimpleNamespace(
        cfg=str(ROOT / "unet/configs/swin_tiny_patch4_window7_224_lite.yaml"),
        opts=None, batch_size=1, img_size=224, zip=False, cache_mode="part",
        resume=None, accumulation_steps=None, use_checkpoint=False,
        amp_opt_level="O1", tag=None, eval=False, throughput=False,
    )
    config = module.get_config(args)
    assert Path(config.MODEL.PRETRAIN_CKPT).is_absolute()
    assert Path(config.MODEL.PRETRAIN_CKPT) == ROOT / "unet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth"


def test_yolo_export_writes_empty_predictions(tmp_path):
    module = load_module("yolo_reproduction_masks", "yolo/predict_masks.py")
    result = SimpleNamespace(path="empty.png", orig_shape=(8, 12), masks=None)
    module.save_result_mask(result, tmp_path)
    mask = np.asarray(Image.open(tmp_path / "empty.png"))
    assert mask.shape == (8, 12)
    assert not mask.any()


def test_yolo_export_combines_instances_at_original_resolution(tmp_path):
    import torch
    module = load_module("yolo_reproduction_masks", "yolo/predict_masks.py")
    data = torch.zeros((2, 8, 12))
    data[0, 1:3, 2:4] = 1
    data[1, 5:7, 8:10] = 1
    result = SimpleNamespace(path="two.png", orig_shape=(8, 12), masks=SimpleNamespace(data=data))
    module.save_result_mask(result, tmp_path)
    mask = np.asarray(Image.open(tmp_path / "two.png"))
    assert mask.shape == (8, 12)
    assert (mask > 0).sum() == 8


def test_bundled_yolo_accepts_standard_training_yaml(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import MODES
    from ultralytics.data import utils
    import yaml

    assert "train" in MODES
    for name in ["train", "val", "test"]:
        (tmp_path / name).mkdir()
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "train",
                                    "val": "val", "test": "test", "names": ["defect"]}),
                      encoding="utf-8")
    monkeypatch.setattr(utils, "check_font", lambda *args, **kwargs: None)
    data = utils.check_det_dataset(str(config), autodownload=False)
    assert Path(data["train"]) == tmp_path / "train"


def test_sam_annotation_image_loader_preserves_unicode_and_bgr(tmp_path):
    module = load_module("sam_reproduction_utils", "sam/SAM-Tool/salt/utils.py")
    path = tmp_path / "图像.png"
    Image.fromarray(np.full((4, 4, 3), [20, 40, 60], np.uint8)).save(path)
    image = module.read_image(path)
    assert image.shape == (4, 4, 3)
    assert image[0, 0].tolist() == [60, 40, 20]
