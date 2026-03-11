import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from sam2.utils.transforms import SAM2Transforms
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock

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


def show_mask(ax, mask, alpha=0.5):
    color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
    h, w = mask.shape
    overlay = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(overlay)


class SAM2ImagePredictor(torch.nn.Module):
    def __init__(self, sam_model, transforms, mask_threshold=0.0):
        super().__init__()
        self.model = sam_model
        self.device = self.model.device

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self._transforms = transforms

    def forward(self, image, org_hw, point_coords, point_labels):
        self._orig_hw = [org_hw]
        input_image = image[None, ...].to(self.device)
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, None, None, True
        )

        return self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            None,
            True,
            return_logits=False,
        )

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords,
        point_labels,
        boxes=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
        img_idx=-1,
    ):
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        return low_res_masks, iou_predictions

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]


if __name__ == "__main__":

    CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
    IMAGE_PATH = "truck.jpg"

    # Foreground point on the truck
    POINT_COORDS = torch.tensor([[500, 375]], dtype=torch.float)
    POINT_LABELS = torch.tensor([1])

    image = Image.open(IMAGE_PATH).convert("RGB")
    org_size = image.size[::-1]

    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd)
    model = model.to("cpu").eval()
    transforms = SAM2Transforms(
        resolution=model.image_size,
        mask_threshold=0,
        max_hole_area=0,
        max_sprinkle_area=0,
    )
    image_tensor = transforms(image)

    predictor = SAM2ImagePredictor(model, transforms)

    with torch.no_grad():
        args = (
            image_tensor,
            torch.tensor(org_size),
            POINT_COORDS,
            POINT_LABELS,
        )
        traced = torch.jit.trace(predictor, args)
        low_res_masks, ious = traced(*args)
        masks = transforms.postprocess_masks(low_res_masks, org_size)
        masks = (masks > 0.0).squeeze(0).float().detach().numpy()
        ious = ious.squeeze(0).detach().numpy()

    fig, axes = plt.subplots(1, masks.shape[0], figsize=(5 * masks.shape[0], 5))
    for i, ax in enumerate(axes):
        ax.imshow(image)
        show_mask(ax, masks[i])
        ax.plot(*POINT_COORDS[0], "r*", markersize=12)
        ax.set_title(f"Mask {i}  IoU={ious[i]:.2f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
