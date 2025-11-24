# Local Inference Demo

Run lightweight forward passes for the bundled architectures without pulling any external assets.

## Setup

- From the repo root, create a Poetry environment (no dependencies are pulled beyond your base interpreter):
  ```bash
  poetry install --no-root
  ```
- Ensure runtime dependencies already exist in your environment: PyTorch (for NAFNet and SAM2), TensorFlow (for RetinaFace), NumPy, Hydra/OmegaConf, and Pillow.

## Usage

- Run every demo on CPU:
  ```bash
  poetry run python inference_demo.py --model all --device cpu
  ```
- Run a single model:
  ```bash
  poetry run inference-demo --model nafnet --device cpu
  ```

Each demo feeds synthetic inputs through the architecture to prove the inference path works without downloading checkpoints or datasets. SAM2 uses the `configs/sam2/sam2_hiera_t.yaml` configuration with random initialization; RetinaFace uses the backbone only (no weight download).
