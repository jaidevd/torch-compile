from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io

from models import RetinaFace
from utils.box_utils import decode, decode_landmarks
from torchvision.ops import nms
from utils.general import draw_detections

RGB_MEAN = (104, 117, 123)
DEFAULT_IMAGE = "assets/test.jpg"
DEFAULT_WEIGHTS = os.path.join("weights", "retinaface_mv2.pth")
CONFIG = {
    "name": "mobilenet_v2",
    "min_sizes": torch.tensor([[16, 32], [64, 128], [256, 512]]),
    "steps": torch.tensor([8, 16, 32]),
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
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_image(image_path: str, device: torch.device):
    """Load and normalize an image for RetinaFace inference.

    Returns the original image, the normalized model input tensor, the original
    height/width, and the padded square size used before resizing.
    """
    original_image = io.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    image = np.float32(original_image)
    img_height, img_width, _ = image.shape

    # Match the training pipeline: pad to square with mean value and resize to model size
    long_side = max(img_height, img_width)
    padded = np.empty((long_side, long_side, 3), dtype=np.float32)
    padded[:] = RGB_MEAN
    padded[:img_height, :img_width] = image

    resized = cv2.resize(padded, (CONFIG["image_size"], CONFIG["image_size"]))
    resized -= RGB_MEAN
    resized = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)

    return original_image, resized, img_height, img_width, long_side


class FaceDetector(torch.nn.Module):

    def __init__(
        self,
        config,
        conf_threshold=0.02,
        nms_threshold=0.4,
        pre_nms_topk=5000,
        post_nms_topk=750,
    ):
        super().__init__()
        self.config = config
        self.device = get_device()
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.retinaface = RetinaFace(cfg=config).to(self.device)
        state_dict = torch.load(
            "weights/retinaface_mv2.pth", map_location=self.device, weights_only=True
        )
        self.retinaface.load_state_dict(state_dict)
        self.min_sizes = torch.tensor([[16, 32], [64, 128], [256, 512]])

    def forward(self, image, image_wh):
        loc, conf, landmarks = self.retinaface(image)
        loc = loc.squeeze(0)
        conf = conf.squeeze(0)
        landmarks = landmarks.squeeze(0)

        image_size = torch.flip(image_wh, (0,))
        steps = self.config["steps"]
        feature_maps = image_size.tile(steps.shape[0], 1) / steps.reshape(-1, 1)
        feature_maps = torch.ceil(feature_maps).int()

        priors = []
        priors.append(_generate_anchors(80, 80, self.min_sizes[0], image_size, 8))
        priors.append(_generate_anchors(40, 40, self.min_sizes[1], image_size, 16))
        priors.append(_generate_anchors(20, 20, self.min_sizes[2], image_size, 32))
        priors = torch.vstack(priors).to(self.device)

        boxes = decode(loc, priors, self.config["variance"])
        landmarks = decode_landmarks(landmarks, priors, self.config["variance"])

        image_wh = image_wh.to(self.device)
        boxes = boxes * torch.tile(image_wh, (1, 2))

        landmarks = landmarks * torch.tile(image_wh, (1, 5))

        scores = conf[:, 1]
        inds = scores > self.conf_threshold
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        order = scores.argsort(descending=True)[: self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        keep = nms(boxes, scores, self.nms_threshold)[: self.post_nms_topk]

        detections = boxes[keep]
        landmarks = landmarks[keep]
        scores = scores[keep]

        return torch.cat((detections, scores.reshape(-1, 1), landmarks), axis=1)


def _generate_anchors(map_width, map_height, min_size, image_size, step):
    xx = (torch.arange(map_width) + 0.5) * step
    yy = (torch.arange(map_height) + 0.5) * step

    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    zz = torch.stack((xx.ravel(), yy.ravel()), dim=1)
    zz = torch.repeat_interleave(zz, 2, dim=0) / image_size.flip(0)

    scaled_sizes = min_size.reshape(1, -1) / image_size.reshape(-1, 1)
    skxy = scaled_sizes.flipud().T.tile(map_height * map_width, 1)
    return torch.hstack((zz, skxy))


def _rescale_detections(detections, img_width, img_height, padded_size, model_input_size):
    """Map detections from model input space back to the original image size."""
    if detections.size == 0:
        return detections

    detections = detections.copy()
    scale = padded_size / model_input_size

    boxes = detections[:, :4] * scale
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, img_width - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, img_height - 1)

    landmarks = detections[:, 5:] * scale
    landmarks = landmarks.reshape(-1, 5, 2)
    landmarks[..., 0] = landmarks[..., 0].clip(0, img_width - 1)
    landmarks[..., 1] = landmarks[..., 1].clip(0, img_height - 1)

    detections[:, :4] = boxes
    detections[:, 5:] = landmarks.reshape(-1, 10)
    return detections


def trace():
    torch_device = get_device()
    _, image_tensor, _, _, _ = prepare_image(
        DEFAULT_IMAGE, torch_device
    )
    model = FaceDetector(config=CONFIG).eval()
    model_input_size = CONFIG["image_size"]
    return torch.jit.trace(model, (image_tensor, torch.tensor((model_input_size, model_input_size))))


def main(
    model=None,
    vis_threshold: float = 0.6,
    conf_threshold: float = 0.02,
    nms_threshold: float = 0.4,
):
    """Detect faces in the provided image and visualize bounding boxes."""
    torch_device = get_device()

    original_image, image_tensor, img_height, img_width, padded_size = prepare_image(
        DEFAULT_IMAGE, torch_device
    )
    model_input_size = CONFIG["image_size"]
    if model is None:
        model = FaceDetector(
            config=CONFIG,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        ).eval()
    with torch.no_grad():
        detections = (
            model(image_tensor, torch.tensor((model_input_size, model_input_size))).cpu().numpy()
        )

    annotated = original_image.copy()
    detections = _rescale_detections(
        detections,
        img_width=img_width,
        img_height=img_height,
        padded_size=padded_size,
        model_input_size=model_input_size,
    )

    draw_detections(annotated, detections, vis_threshold)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(annotated_rgb)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    traced = trace()
    main(traced)
