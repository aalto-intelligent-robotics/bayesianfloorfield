from pathlib import Path

import yaml
from PIL import Image


class OccupancyMap:
    def __init__(
        self,
        image_file,
        resolution,
        origin,
        negate,
        occupied_thresh,
        free_thresh,
    ):
        self.resolution = resolution
        self.origin = origin
        self.negate = negate
        self.occupied_thresh = (
            round(occupied_thresh * 255)
            if occupied_thresh < 1
            else round(occupied_thresh)
        )
        self.free_thresh = (
            round(free_thresh * 255) if free_thresh < 1 else round(free_thresh)
        )
        self.map = Image.open(image_file)

    @property
    def binary_map(self):
        if self.negate:
            return self.map.point(lambda p: p > self.occupied_thresh and 255)
        else:
            return self.map.point(
                lambda p: 255 - p > self.occupied_thresh and 255
            )

    @classmethod
    def from_metadata(cls, metadata):
        return cls(
            image_file=metadata["image"],
            resolution=metadata["resolution"],
            origin=metadata["origin"],
            negate=metadata["negate"],
            occupied_thresh=metadata["occupied_thresh"],
            free_thresh=metadata["free_thresh"],
        )

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as stream:
            meta = yaml.safe_load(stream)
        if not Path(meta["image"]).is_absolute():
            meta["image"] = (Path(yaml_path).parent / meta["image"]).as_posix()
        return cls.from_metadata(meta)
