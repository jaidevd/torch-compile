"""
Lightweight inference demonstrations for the local NAFNet, RetinaFace, and SAM2 sources.

The script keeps everything self contained by importing the checked-in packages
directly from this repository. It runs minimal forward passes on synthetic data
to validate that inference code paths are wired correctly without downloading
weights or assets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Make the local projects importable without installing them.
ROOT = Path(__file__).resolve().parent
for project_dir in (ROOT / "NAFNet", ROOT / "retinaface", ROOT / "sam2"):
    sys.path.append(str(project_dir))


def run_nafnet(device: str = "cpu") -> Dict[str, Any]:
    """Run a tiny NAFNet forward pass on synthetic data."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run the NAFNet demo.") from exc

    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Failed to import NAFNet from the local basicsr package.") from exc

    model = NAFNet(
        img_channel=3,
        width=8,
        middle_blk_num=1,
        enc_blk_nums=[1, 1],
        dec_blk_nums=[1, 1],
    ).to(device)
    model.eval()

    dummy = torch.randn(1, 3, 64, 64, device=device)
    with torch.inference_mode():
        output = model(dummy)

    return {
        "model": "NAFNet",
        "device": device,
        "input_shape": tuple(dummy.shape),
        "output_shape": tuple(output.shape),
    }


def run_retinaface(device: str = "cpu") -> Dict[str, Any]:
    """Run the Torch-based RetinaFace backbone on synthetic data."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run the RetinaFace demo.") from exc

    try:
        from retinaface.config import get_config
        from retinaface.models import RetinaFace
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Failed to import RetinaFace from the local source tree.") from exc

    cfg = get_config("mobilenetv2")
    if cfg is None:
        raise RuntimeError("RetinaFace config mobilenetv2 not found.")
    cfg = {**cfg, "pretrain": False}  # keep offline

    model = RetinaFace(cfg=cfg).to(device)
    model.eval()

    dummy = torch.randn(1, 3, cfg.get("image_size", 640), cfg.get("image_size", 640), device=device)
    with torch.inference_mode():
        loc, conf, landmarks = model(dummy)

    return {
        "model": "RetinaFace (mobilenetv2, random init)",
        "device": device,
        "input_shape": tuple(dummy.shape),
        "loc_shape": tuple(loc.shape),
        "conf_shape": tuple(conf.shape),
        "landmark_shape": tuple(landmarks.shape),
    }


def run_sam2(device: str = "cpu") -> Dict[str, Any]:
    """
    Run a SAM2 image predictor pass on a synthetic image.

    We keep the config small (hiera tiny) and skip checkpoints so everything
    stays offline.
    """
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch and NumPy are required to run the SAM2 demo.") from exc

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Failed to import SAM2 from the local source tree.") from exc

    model_cfg = "configs/sam2/sam2_hiera_t.yaml"
    predictor = SAM2ImagePredictor(
        build_sam2(
            config_file=model_cfg,
            ckpt_path=None,
            device=device,
            mode="eval",
            apply_postprocessing=False,
        )
    )

    synthetic_image = np.zeros((128, 128, 3), dtype=np.uint8)
    predictor.set_image(synthetic_image)
    masks, ious, low_res_masks = predictor.predict(
        point_coords=np.array([[64, 64]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=False,
    )

    return {
        "model": "SAM2 (hiera tiny cfg, random init)",
        "device": device,
        "image_shape": synthetic_image.shape,
        "mask_shape": tuple(masks.shape),
        "iou_shape": tuple(ious.shape),
        "low_res_shape": tuple(low_res_masks.shape),
    }


def _select_models(choice: str) -> Iterable[str]:
    if choice == "all":
        return ("nafnet", "retinaface", "sam2")
    return (choice,)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local inference demos.")
    parser.add_argument(
        "--model",
        choices=["all", "nafnet", "retinaface", "sam2"],
        default="all",
        help="Which model demo to run.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for NAFNet and SAM2 (e.g. cpu, cuda, mps).",
    )
    args = parser.parse_args(argv)

    runners = {
        "nafnet": lambda: run_nafnet(device=args.device),
        "retinaface": lambda: run_retinaface(device=args.device),
        "sam2": lambda: run_sam2(device=args.device),
    }

    for key in _select_models(args.model):
        print(f"=== Running {key} demo ===")
        result = runners[key]()
        for field, value in result.items():
            print(f"{field}: {value}")
        print()


if __name__ == "__main__":
    main()
