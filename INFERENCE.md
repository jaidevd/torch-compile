# Local Inference Tests

Each model directory now has its own lightweight inference script that drives a synthetic forward pass without downloading external assets. Run them independently so you can use separate Python environments per project.

- NAFNet: `cd NAFNet && python inference.py --device cpu`
- RetinaFace: `cd retinaface && python inference.py --device cpu`
- SAM2: `cd sam2 && python inference.py --device cpu`

These scripts validate that the inference code paths are wired correctly with random initialization. SAM2 uses the `sam2/configs/sam2/sam2_hiera_t.yaml` configuration; RetinaFace uses the mobilenetv2 backbone only (no weight download). Ensure the usual runtime dependencies (PyTorch/NumPy/etc.) are available in whichever interpreter you use for each project.
