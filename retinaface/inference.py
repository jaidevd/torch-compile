from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io

from layers import PriorBox
from models import RetinaFace
from utils.box_utils import decode, decode_landmarks
from torchvision.ops import nms
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


class FaceDetector(torch.nn.Module):

    def __init__(self, config, conf_threshold=0.02, nms_threshold=0.4, pre_nms_topk=5000, post_nms_topk=750):
        super().__init__()
        self.config = config
        self.device = get_device()
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.retinaface = RetinaFace(cfg=config).to(self.device)
        state_dict = torch.load(
            "weights/retinaface_mv2.pth", map_location=self.device, weights_only=True)
        self.retinaface.load_state_dict(state_dict)

    def forward(self, image, image_wh):
        loc, conf, landmarks = self.retinaface(image)
        loc = loc.squeeze(0)
        conf = conf.squeeze(0)
        landmarks = landmarks.squeeze(0)

        priorbox = PriorBox(self.config, image_size=image_wh[::-1])
        priors = priorbox.generate_anchors().to(self.device)

        boxes = decode(loc, priors, self.config["variance"])
        landmarks = decode_landmarks(landmarks, priors, self.config["variance"])

        bbox_scale = torch.tensor(image_wh * 2, device=self.device)
        boxes = boxes * bbox_scale

        landmark_scale = torch.tensor(image_wh * 5, device=self.device)
        landmarks = (landmarks * landmark_scale)

        scores = conf[:, 1]
        inds = scores > self.conf_threshold
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        order = scores.argsort(descending=True)[:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        keep = nms(boxes, scores, self.nms_threshold)[:self.post_nms_topk]

        detections = boxes[keep]
        landmarks = landmarks[keep]
        scores = scores[keep]

        return torch.cat((detections, scores.reshape(-1, 1), landmarks), axis=1)


def main(
    vis_threshold: float = 0.6,
    conf_threshold: float = 0.02,
    nms_threshold: float = 0.4,
):
    """Detect faces in the provided image and visualize bounding boxes."""
    torch_device = get_device()

    original_image, image_tensor, img_height, img_width = prepare_image(
        DEFAULT_IMAGE, torch_device
    )

    with torch.no_grad():
        model = FaceDetector(
            config=CONFIG,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        ).eval()
        detections = model(image_tensor, (img_width, img_height)).cpu().numpy()

    annotated = original_image.copy()

    draw_detections(annotated, detections, vis_threshold)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(annotated_rgb)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
