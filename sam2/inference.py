from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
IMAGE_PATH = "truck.jpg"

# Foreground point on the truck
POINT_COORDS = np.array([[500, 375]], dtype=np.float32)
POINT_LABELS = np.array([1], dtype=np.int32)


def show_mask(ax, mask, alpha=0.5):
    color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
    h, w = mask.shape
    overlay = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(overlay)


image = np.array(Image.open(IMAGE_PATH).convert("RGB"))


predictor = SAM2ImagePredictor(
    build_sam2(MODEL_CFG, ckpt_path=CHECKPOINT, device="cpu", mode="eval",
               apply_postprocessing=True)
)
predictor.set_image(image)
masks, ious, _ = predictor.predict(
    point_coords=POINT_COORDS,
    point_labels=POINT_LABELS,
    multimask_output=True,
)

fig, axes = plt.subplots(1, masks.shape[0], figsize=(5 * masks.shape[0], 5))
for i, ax in enumerate(axes):
    ax.imshow(image)
    show_mask(ax, masks[i])
    ax.plot(*POINT_COORDS[0], "r*", markersize=12)
    ax.set_title(f"Mask {i}  IoU={ious[i]:.2f}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("output.png", dpi=150)
plt.show()
