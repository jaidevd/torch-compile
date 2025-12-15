from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def _img2tensor(img: np.ndarray, bgr2rgb: bool = False, float32: bool = True):
    from basicsr.utils import img2tensor as _img2tensor

    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def _run_single_image(
    opt_path: Path, img_path: Path, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    from basicsr.models import create_model
    from basicsr.utils.options import parse
    from basicsr.utils import tensor2img

    opt = parse(str(opt_path), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 1 if device.startswith("cuda") else 0

    import torch

    model = create_model(opt)
    model.device = torch.device(device)
    model.net_g.to(model.device)
    model.net_g.eval()

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    tensor = _img2tensor(img).unsqueeze(0).to(model.device)

    model.feed_data(data={"lq": tensor})
    if model.opt["val"].get("grids", False):
        model.grids()
    model.test()
    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    restored = tensor2img([visuals["result"]])
    return img, restored


def _run_stereo_image(
    opt_path: Path, img_l_path: Path, img_r_path: Path, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from basicsr.models import create_model
    from basicsr.utils.options import parse
    from basicsr.utils import tensor2img

    opt = parse(str(opt_path), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 1 if device.startswith("cuda") else 0

    import torch

    model = create_model(opt)
    model.device = torch.device(device)
    model.net_g.to(model.device)
    model.net_g.eval()

    img_l = cv2.cvtColor(cv2.imread(str(img_l_path)), cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(cv2.imread(str(img_r_path)), cv2.COLOR_BGR2RGB)
    tensor_l = _img2tensor(img_l).unsqueeze(0).to(model.device)
    tensor_r = _img2tensor(img_r).unsqueeze(0).to(model.device)
    tensor = torch.cat([tensor_l, tensor_r], dim=1)
    model.feed_data(data={"lq": tensor})

    if model.opt["val"].get("grids", False):
        model.grids()
    model.test()
    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    result = visuals["result"]
    img_L = result[:, :3]
    img_R = result[:, 3:]
    sr_l, sr_r = tensor2img([img_L, img_R], rgb2bgr=False)
    return img_l, img_r, sr_l, sr_r


def _show_images(
    images: Tuple[np.ndarray, ...],
    titles: Tuple[str, ...],
    cols: int,
    figsize: Tuple[int, int],
) -> None:
    rows = math.ceil(len(images) / cols)
    plt.figure(figsize=figsize)
    for idx, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(rows, cols, idx)
        plt.title(title)
        plt.axis("off")
        plt.imshow(img)
    plt.tight_layout()
    plt.show()


def run_all(device: str = "cpu") -> Dict[str, str]:
    base = Path(__file__).resolve().parent

    denoise_opt = base / "options/test/SIDD/NAFNet-width64.yml"
    denoise_img = base / "demo/noisy.png"
    deblur_opt = base / "options/test/GoPro/NAFNet-width64.yml"
    deblur_img = base / "demo/blurry.jpg"
    stereo_opt = base / "options/test/NAFSSR/NAFSSR-L_4x.yml"
    stereo_l = base / "demo/lr_img_l.png"
    stereo_r = base / "demo/lr_img_r.png"

    # Denoise
    orig, restored = _run_single_image(denoise_opt, denoise_img, device=device)
    _show_images(
        images=(orig, restored),
        titles=("Noisy input", "Denoised (NAFNet)"),
        cols=2,
        figsize=(10, 4),
    )

    # Deblur
    orig, restored = _run_single_image(deblur_opt, deblur_img, device=device)
    _show_images(
        images=(orig, restored),
        titles=("Blurry input", "Deblurred (NAFNet)"),
        cols=2,
        figsize=(10, 4),
    )

    # Stereo SR
    orig_l, orig_r, sr_l, sr_r = _run_stereo_image(
        stereo_opt, stereo_l, stereo_r, device=device)
    _show_images(
        images=(orig_l, orig_r, sr_l, sr_r),
        titles=(
            "Stereo Left (LR)",
            "Stereo Right (LR)",
            "Stereo Left SR (NAFSSR)",
            "Stereo Right SR (NAFSSR)",
        ),
        cols=2,
        figsize=(12, 8),
    )


if __name__ == "__main__":
    run_all(device="mps")
