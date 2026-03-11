import os
import cv2
import argparse
import numpy as np

import torch

from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.general import draw_detections
from utils.box_utils import decode, decode_landmarks, nms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Retinaface Webcam Inference")

    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/retinaface_mv2.pth',
        help='Path to the trained model weights'
    )
    parser.add_argument(
        '-n', '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )

    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.4,
        help='Confidence threshold for filtering detections'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=5000,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=750,
        help='Number of highest scoring detections to keep after NMS'
    )

    # Output options
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=0.6,
        help='Visualization threshold for displaying detections'
    )
    
    # Video saving options
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Input video path or Webcam source (default: 0)'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Enable saving the processed video'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./output_video.mp4',
        help='Path to save the output video'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=24.0,
        help='FPS for the output video'
    )

    return parser.parse_args()


@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf, landmarks = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    return loc, conf, landmarks


def resize_image(frame, target_shape=(640, 640)):
    width, height = target_shape

    # Aspect-ratio preserving resize
    im_ratio = float(frame.shape[0]) / frame.shape[1]
    model_ratio = height / width
    if im_ratio > model_ratio:
        new_height = height
        new_width = int(new_height / im_ratio)
    else:
        new_width = width
        new_height = int(new_width * im_ratio)

    resize_factor = float(new_height) / frame.shape[0]
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create blank image and place resized image on it
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:new_height, :new_width, :] = resized_frame

    return image, resize_factor


def main(params):
    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (104, 117, 123)
    resize_factor = 1

    # model initialization
    model = RetinaFace(cfg=cfg)
    model.to(device)

    # loading state_dict
    state_dict = torch.load(params.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # Open webcam
    if params.source.isdigit():
        cap = cv2.VideoCapture(int(params.source))
    else:
        cap = cv2.VideoCapture(params.source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties for output writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer if save option is enabled
    video_writer = None
    if params.save_video:
        # Ensure output directory exists
        output_dir = os.path.dirname(params.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        video_writer = cv2.VideoWriter(
            params.output_path,
            fourcc,
            params.fps,
            (frame_width, frame_height)
        )
        print(f"Video will be saved to: {params.output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        image, resize_factor = resize_image(frame, target_shape=(640, 640))

        # Prepare image for inference
        image = np.float32(image)
        img_height, img_width, _ = image.shape
        image -= rgb_mean
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        # forward pass
        loc, conf, landmarks = inference(model, image)

        # generate anchor boxes
        priorbox = PriorBox(cfg, image_size=(img_height, img_width))
        priors = priorbox.generate_anchors().to(device)

        # decode boxes and landmarks
        boxes = decode(loc, priors, cfg['variance'])
        landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

        # scale adjustments
        bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
        boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()

        landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
        landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

        scores = conf.cpu().numpy()[:, 1]

        # filter by confidence threshold
        inds = scores > params.conf_threshold
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        # sort by scores
        order = scores.argsort()[::-1][:params.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, params.nms_threshold)

        detections = detections[keep]
        landmarks = landmarks[keep]

        # keep top-k detections and landmarks
        detections = detections[:params.post_nms_topk]
        landmarks = landmarks[:params.post_nms_topk]

        # concatenate detections and landmarks
        detections = np.concatenate((detections, landmarks), axis=1)

        # draw detections on the frame
        draw_detections(frame, detections, params.vis_threshold)

        # Write frame to output video if enabled
        if params.save_video and video_writer is not None:
            video_writer.write(frame)

        # Display the resulting frame
        cv2.imshow('Webcam Inference', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    if params.save_video:
        print(f"Video saved successfully to {params.output_path}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
