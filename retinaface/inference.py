from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io

from layers import PriorBox
from models import RetinaFace
from utils.box_utils import decode, decode_landmarks, nms
from utils.general import draw_detections

RGB_MEAN = (104, 117, 123)
DEFAULT_IMAGE = os.path.join("assets", "test.jpg")
DEFAULT_WEIGHTS = os.path.join("weights", "retinaface_mv2.pth")
CONFIG = {
    "name": "mobilenet_v2",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "batch_size": 32,
    "epochs": 250,
    "milestones": [190, 220],
    "image_size": 640,
    "pretrain": True,
    "return_layers": [6, 13, 18],
    "in_channel": 32,
    "out_channel": 128,
}


def get_device() -> torch.device:
    """Resolve a torch device string, preferring MPS then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    """Load a RetinaFace model and its config."""
    device = get_device()
    model = RetinaFace(cfg=CONFIG).to(device)
    state_dict = torch.load(
        "weights/retinaface_mv2.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_image(image_path: str, device: torch.device):
    """Load and normalize an image for RetinaFace inference."""
    original_image = io.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    image = np.float32(original_image)
    img_height, img_width, _ = image.shape

    image -= RGB_MEAN
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    return original_image, image, img_height, img_width


def detect_faces(
    model: RetinaFace,
    image: torch.Tensor,
    img_height: int,
    img_width: int,
    device: torch.device,
    conf_threshold: float = 0.02,
    nms_threshold: float = 0.4,
    pre_nms_topk: int = 5000,
    post_nms_topk: int = 750,
) -> np.ndarray:
    """Run RetinaFace on an image tensor and return decoded detections."""
    with torch.no_grad():
        loc, conf, landmarks = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    priorbox = PriorBox(CONFIG, image_size=(img_height, img_width))
    priors = priorbox.generate_anchors().to(device)

    boxes = decode(loc, priors, CONFIG["variance"])
    landmarks = decode_landmarks(landmarks, priors, CONFIG["variance"])

    bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
    boxes = (boxes * bbox_scale).cpu().numpy()

    landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
    landmarks = (landmarks * landmark_scale).cpu().numpy()

    scores = conf.cpu().numpy()[:, 1]

    inds = scores > conf_threshold
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1][:pre_nms_topk]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    detections = np.hstack(
        (boxes, scores[:, np.newaxis])
    ).astype(np.float32, copy=False)
    keep = nms(detections, nms_threshold)

    detections = detections[keep]
    landmarks = landmarks[keep]

    detections = detections[:post_nms_topk]
    landmarks = landmarks[:post_nms_topk]

    return np.concatenate((detections, landmarks), axis=1)


def main(
    vis_threshold: float = 0.6,
    conf_threshold: float = 0.02,
    nms_threshold: float = 0.4,
):
    """Detect faces in the provided image and visualize bounding boxes."""
    torch_device = get_device()

    model = load_model()
    original_image, image_tensor, img_height, img_width = prepare_image(
        DEFAULT_IMAGE, torch_device
    )

    # Tracing
    with torch.no_grad():
        traced_model = torch.jit.trace(model, image_tensor)

    detections = detect_faces(
        traced_model,
        image_tensor,
        img_height,
        img_width,
        torch_device,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )

    annotated = original_image.copy()
    draw_detections(annotated, detections, vis_threshold)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(annotated_rgb)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
