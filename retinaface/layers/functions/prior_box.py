import torch
from typing import Tuple


class PriorBox:
    def __init__(self, cfg: dict, image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.image_size = image_size
        self.clip = cfg['clip']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        feature_maps = image_size.tile(
            self.steps.shape[0], 1
        ) / self.steps.reshape(-1, 1)
        self.feature_maps = torch.ceil(feature_maps).int()

    def generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes based on configuration and image size"""
        t_anchors = []
        map_heights, map_widths = self.feature_maps.T
        height, width = self.image_size
        for k, (map_height, map_width) in enumerate(zip(map_heights, map_widths)):
            step = self.steps[k]

            xx = (torch.arange(map_width) + 0.5) * step / width
            yy = (torch.arange(map_height) + 0.5) * step / height
            yy, xx = torch.meshgrid(yy, xx, indexing='ij')
            zz = torch.stack((xx.ravel(), yy.ravel()), dim=1)
            zz = torch.repeat_interleave(zz, 2, dim=0)

            s_kx = self.min_sizes[k] / width
            s_ky = self.min_sizes[k] / height
            skxy = torch.vstack((s_kx, s_ky)).T.tile(int(map_height * map_width), 1)
            t_anchors.append(torch.hstack((zz, skxy)))

        # back to torch land
        t_output = torch.vstack(t_anchors)
        if self.clip:
            t_output.clamp_(max=1, min=0)
        return t_output
