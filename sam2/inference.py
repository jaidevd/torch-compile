from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
MODEL_CFG = ROOT / "sam2" / "configs" / "sam2" / "sam2_hiera_t.yaml"


def run(device: str = "cpu") -> Dict[str, Any]:
    """
    Run a SAM2 image predictor pass on a synthetic image using the tiny config.
    """
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch and NumPy are required to run the SAM2 inference test.") from exc

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Failed to import SAM2 from the local source tree.") from exc

    if not MODEL_CFG.is_file():
        raise RuntimeError(f"SAM2 config not found at {MODEL_CFG}")

    predictor = SAM2ImagePredictor(
        build_sam2(
            config_file=str(MODEL_CFG),
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
        "config": str(MODEL_CFG),
        "image_shape": synthetic_image.shape,
        "mask_shape": tuple(masks.shape),
        "iou_shape": tuple(ious.shape),
        "low_res_shape": tuple(low_res_masks.shape),
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a local SAM2 inference test.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (e.g. cpu, cuda, mps).",
    )
    args = parser.parse_args(argv)

    result = run(device=args.device)
    for field, value in result.items():
        print(f"{field}: {value}")


if __name__ == "__main__":
    main()
