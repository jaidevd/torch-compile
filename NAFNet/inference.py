from __future__ import annotations

import argparse
from typing import Any, Dict, List


def run(device: str = "cpu") -> Dict[str, Any]:
    """Run a tiny NAFNet forward pass on synthetic data."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run the NAFNet inference test.") from exc

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


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a local NAFNet inference test.")
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
