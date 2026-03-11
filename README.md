# Making Real-World Models `torch.compile`-Ready

A hands-on workshop on making PyTorch models traceable. We work through three models of increasing complexity: **RetinaFace** (face detection), **NAFNet** (image restoration), and **SAM2** (segmentation).

## Prerequisites

- Python 3.11
- Git

## Setup

Clone the repository:

```bash
git clone <repo-url> torch-compile
cd torch-compile
```

Each model has its own `pyproject.toml`. Install dependencies per model with uv (recommended):

```bash
cd retinaface && uv sync && cd ..
cd NAFNet && uv sync && cd ..
cd sam2 && uv sync && cd ..
```

Or with pip:

```bash
pip install -e retinaface/
pip install -e NAFNet/
pip install -e sam2/
```

## Download Model Weights

Each model needs pre-trained weights. Download them before the workshop.

### RetinaFace

```bash
cd retinaface/weights
wget https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv2.pth
cd ../..
```

### NAFNet

Download the denoising and stereo super-resolution checkpoints into `NAFNet/experiments/pretrained_models/`:

```bash
mkdir -p NAFNet/experiments/pretrained_models
```

| Model | Google Drive |
|---|---|
| NAFNet-SIDD-width64.pth | https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view?usp=sharing |
| NAFSSR-L_4x.pth | https://drive.google.com/file/d/1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb/view?usp=sharing |

Download both files and place them in `NAFNet/experiments/pretrained_models/`.

### SAM2

```bash
cd sam2/checkpoints
bash download_ckpts.sh
cd ../..
```

Or download just the tiny model (recommended for the workshop):

```bash
wget -P sam2/checkpoints \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

## Verify Your Setup

Run each inference script to confirm everything works:

```bash
# RetinaFace
cd retinaface
python inference.py
cd ..

# NAFNet - denoising
cd NAFNet
python denoise.py
cd ..

# NAFNet - stereo super-resolution
cd NAFNet
python ssr.py
cd ..

# SAM2
cd sam2
python inference.py
cd ..
```

Each script should open a matplotlib window showing the model's output. Close the window to continue.

## Workshop Flow

We work through each model in order. The goal for each: make the model fully traceable with `torch.jit.trace`, eliminating all `TracerWarning`s.

1. **RetinaFace** -- The model traces fine; the postprocessing does not. We vectorize anchor generation, fix type collisions (numpy vs torch, Python floats vs tensors), and standardize input sizes.

2. **NAFNet** -- Denoising traces immediately (nothing to do). Stereo super-resolution hits familiar patterns: tensor-to-boolean, vector unpacking as iteration, creating tensors from shapes. Fixes come faster this time.

3. **SAM2** -- The most complex model. We flatten the architecture (remove Hydra, collapse the predictor), make it stateless, reduce to a single `.forward()`, then work through the remaining tracer warnings one by one.
