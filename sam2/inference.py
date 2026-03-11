from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.sam2_image_predictor import SAM2ImagePredictor

CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
IMAGE_PATH = "truck.jpg"

# Foreground point on the truck
POINT_COORDS = np.array([[500, 375]], dtype=np.float32)
POINT_LABELS = np.array([1], dtype=np.int32)


trunk = Hiera(
    embed_dim=96,
    num_heads=1,
    stages=[1, 2, 7, 2],
    global_att_blocks=[5, 7, 9],
    window_pos_embed_bkg_spatial_size=[7, 7],
)
neck = FpnNeck(
    position_encoding=PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    ),
    d_model=256,
    backbone_channel_list=[768, 384, 192, 96],
    fpn_top_down_levels=[2, 3],
    fpn_interp_model="nearest",
)
image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)

# Memory attention
memory_attention = MemoryAttention(
    d_model=256,
    pos_enc_at_input=True,
    layer=MemoryAttentionLayer(
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        self_attention=RoPEAttention(
            rope_theta=10000.0,
            feat_sizes=[64, 64],
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
        ),
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=RoPEAttention(
            rope_theta=10000.0,
            feat_sizes=[64, 64],
            rope_k_repeat=True,
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
            kv_in_dim=64,
        ),
    ),
    num_layers=4,
)

# Memory encoder
memory_encoder = MemoryEncoder(
    out_dim=64,
    position_encoding=PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
    ),
    mask_downsampler=MaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
    ),
    fuser=Fuser(
        layer=CXBlock(
            dim=256,
            kernel_size=7,
            padding=3,
            layer_scale_init_value=1e-6,
            use_dwconv=True,
        ),
        num_layers=2,
    ),
)

# SAM2 model
model = SAM2Base(
    image_encoder=image_encoder,
    memory_attention=memory_attention,
    memory_encoder=memory_encoder,
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    no_obj_embed_spatial=True,
    use_high_res_features_in_sam=True,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=True,
    proj_tpos_enc_in_obj_ptrs=True,
    use_signed_tpos_enc_to_obj_ptrs=True,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    compile_image_encoder=False,
    sam_mask_decoder_extra_args={
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    },
)
sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model"]
missing_keys, unexpected_keys = model.load_state_dict(sd)

model = model.to("cpu").eval()


def show_mask(ax, mask, alpha=0.5):
    color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
    h, w = mask.shape
    overlay = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(overlay)


image = np.array(Image.open(IMAGE_PATH).convert("RGB"))


predictor = SAM2ImagePredictor(model)
with torch.no_grad():
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
