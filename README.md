# Optimizing PyTorch Models

If you are here because you are attending the workshop at [PyConf Hyderabad 2026](https://2026.pyconfhyd.org/speakers/jaidev-deshpande), then please spare 30 minutes _before_ the workshop to set up this repository on your computer.


## Prerequisites

- [UV](https://docs.astral.sh/uv/#installation)
- Python >=3.11
- Git (with LFS)

## Download pretrained model weights

From the following table, download each file and place it in the specified
location (relative to the repository root). Create folders paths as needed.

| Download Link | Destination Folder |                                                 
|---|---|                                                                              
| [Download](https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv2.pth) | `retinaface/weights/` |
| [Download](https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view?usp=sharing) | `NAFNet/experiments/pretrained_models/` |
| [Download](https://drive.google.com/file/d/1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb/view?usp=sharing) | `NAFNet/experiments/pretrained_models/` |
| [Download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | `sam2/checkpoints/` |

## Setup

The repository contains three architectures: RetinaFace, NAFNet and SAM2. Each
has it's own directory. To finish the setup, navigate into each directory and
setup the dependencies, as follows:


```bash
# RetinaFace
cd retinaface
uv sync && cd ..

# NAFNet
cd NAFNet
uv sync && cd ..

# SAM2
cd sam2
uv sync && cd ..
```

## Testing the setup

Each folder also has its own test script. Please run them one by one as follows:

```bash
# RetinaFace
cd retinaface
uv run python inference.py
cd ..

# NAFNet - denoising & super-stereo-resolution
cd NAFNet
uv run python denoise.py
uv run python ssr.py
cd ..

# SAM2
cd sam2
uv run python inference.py
cd ..
```

**Note**: Each script on a successful run should show you a matplotlib window
with visualizations of the results.

For any help, reach out to me on the official [Discord channel](https://discord.com/channels/1301212203336073226/1475140722394136768) of this workshop and tag me (`@jaidevd`) directly.
