#!/usr/bin/env python3
"""
LithoDefectV1 — One-Click Full Pipeline Runner
===============================================
Implements the complete synthesis pipeline from the paper:
  "Advancing Lithography Process Control via a Synthetic Benchmark
   for Defect Segmentation"

Pipeline steps:
  1. Synthesis  — Generate synthetic defect images + masks
  2. Split      — Stratified train/val/test partition
  3. Export     — Convert to YOLO and Swin-UNet training formats

Usage:
  python run_pipeline.py                        # Run with config.yaml defaults
  python run_pipeline.py --config my_conf.yaml  # Use custom config
  python run_pipeline.py --step synthesis       # Run only synthesis step
"""

import argparse
import os
import sys
import yaml


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def step_synthesis(config):
    """Step 1: Generate synthetic defect dataset."""
    from lithodefect.synthesis import Synthesizer

    syn_config = {
        "defect_img_dir": config["paths"]["defect_img_dir"],
        "defect_mask_dir": config["paths"]["defect_mask_dir"],
        "bg_dir": config["paths"]["bg_dir"],
        "output_root": config["paths"]["output_root"],
        "category_mapping": config["category_mapping"],
        "bg_groups": config["bg_groups"],
        "synthesis_params": config["synthesis_params"],
        "skip_dirs": ["good", "fail"],
    }

    synthesizer = Synthesizer(syn_config)
    synthesizer.run()


def step_split(config):
    """Step 2: Stratified train/val/test split."""
    from lithodefect.split import split_dataset

    split_config = config["split"]
    split_dataset(
        src_root=config["paths"]["output_root"],
        out_root=config["paths"]["split_output"],
        **split_config
    )


def step_export_yolo(config):
    """Step 3a: Export to YOLO segmentation format."""
    from lithodefect.export_yolo import export_to_yolo

    export_to_yolo(
        src_root=config["paths"]["split_output"],
        out_root=config["paths"]["yolo_output"],
        class_mapping=config["class_mapping"],
    )


def step_export_swinunet(config):
    """Step 3b: Export to Swin-UNet format."""
    from lithodefect.export_swinunet import export_to_swinunet

    # Swin-UNet uses class IDs starting from 1 (0=background)
    swin_mapping = {k: v + 1 for k, v in config["class_mapping"].items()}

    export_to_swinunet(
        src_root=config["paths"]["split_output"],
        out_root=config["paths"]["swinunet_output"],
        class_mapping=swin_mapping,
    )


# Step dispatcher
STEPS = {
    "synthesis": ("Synthesis", step_synthesis),
    "split": ("Train/Val/Test Split", step_split),
    "yolo": ("Export YOLO Format", step_export_yolo),
    "swinunet": ("Export Swin-UNet Format", step_export_swinunet),
}

PIPELINE_ORDER = ["synthesis", "split", "yolo", "swinunet"]


def main():
    parser = argparse.ArgumentParser(
        description="LithoDefectV1 — Synthetic Benchmark Pipeline")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")
    parser.add_argument("--step", choices=list(STEPS.keys()) + ["all"],
                        default="all",
                        help="Run a specific pipeline step (default: all)")
    parser.add_argument("--list-steps", action="store_true",
                        help="List available pipeline steps and exit")
    args = parser.parse_args()

    if args.list_steps:
        print("Available pipeline steps:")
        for step_name, (desc, _) in STEPS.items():
            print(f"  {step_name:<12s} — {desc}")
        return

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("Create one by copying and editing config.yaml")
        sys.exit(1)

    config = load_config(args.config)
    print("=" * 60)
    print("  LithoDefectV1 — Synthetic Benchmark Pipeline")
    print("=" * 60)

    pipeline_cfg = config.get("pipeline", {})

    if args.step == "all":
        # Run all enabled steps in order
        for step_name in PIPELINE_ORDER:
            if not pipeline_cfg.get(f"run_{step_name.replace('yolo', 'export_yolo').replace('swinunet', 'export_swinunet')}", True):
                continue
            # Map step names to pipeline cfg keys
            cfg_keys = {
                "synthesis": "run_synthesis",
                "split": "run_split",
                "yolo": "run_export_yolo",
                "swinunet": "run_export_swinunet",
            }
            cfg_key = cfg_keys.get(step_name, f"run_{step_name}")
            if not pipeline_cfg.get(cfg_key, True):
                print(f"\n  [SKIP] Step: {STEPS[step_name][0]}")
                continue

            desc, func = STEPS[step_name]
            print(f"\n{'─' * 60}")
            print(f"  STEP {PIPELINE_ORDER.index(step_name) + 1}/{len(PIPELINE_ORDER)}: {desc}")
            print(f"{'─' * 60}")
            func(config)
    else:
        desc, func = STEPS[args.step]
        print(f"\n  Running single step: {desc}")
        func(config)

    print(f"\n{'=' * 60}")
    print("  Pipeline finished!")
    print(f"  Output directory: {os.path.dirname(config['paths']['output_root'])}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
