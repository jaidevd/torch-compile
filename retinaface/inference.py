from __future__ import annotations

import argparse
from typing import Any, Dict, List


def run(device: str = "cpu") -> Dict[str, Any]:
    """Run the RetinaFace backbone on synthetic data."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run the RetinaFace inference test.") from exc

    try:
        from config import get_config
        from models import RetinaFace
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


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a local RetinaFace inference test.")
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
