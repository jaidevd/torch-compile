from itertools import product

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
        anchors = []
        map_heights, map_widths = self.feature_maps.T
        for k, (map_height, map_width) in enumerate(zip(map_heights, map_widths)):
            step = self.steps[k]
            for i, j in product(range(map_height), range(map_width)):
                for min_size in self.min_sizes[k]:
                    s_kx = min_size / self.image_size[1]
                    dense_cx = (j + 0.5) * step / self.image_size[1]

                    s_ky = min_size / self.image_size[0]
                    dense_cy = (i + 0.5) * step / self.image_size[0]

                    anchors += [dense_cx, dense_cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
