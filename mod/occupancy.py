import logging
from pathlib import Path
from typing import Sequence, Union

import yaml
from PIL import Image

from mod.utils import TDRC_from_XY, TDRCCoords, XY_from_TDRC, XYCoords

logger = logging.getLogger(__name__)


class OccupancyMap:
    """A class representing an occupancy map."""

    def __init__(
        self,
        image_file: Union[str, Path],
        resolution: float,
        origin: Union[XYCoords, Sequence[float]],
        negate: bool,
        occupied_thresh: float,
        free_thresh: float,
    ):
        """Init `OccupancyMap` class.

        Args:
            image_file (Union[str, Path]): path to the occupancy map image.
            resolution (float): resultion of the map in meters per pixel.
            origin (Union[XYCoords, Sequence[float]]): the 2D pose of the
            lower-left pixel in the map, as (x, y), in world coordinates.
            negate (bool): whether the white/black free/occupied semantics
            should be reversed (interpretation of thresholds is unaffected).
            occupied_thresh (float): pixels with occupancy probability greater
            than this threshold are considered completely occupied.
            free_thresh (float): pixels with occupancy probability less than
            this threshold are considered completely free.
        """
        self.resolution = resolution
        self.origin = XYCoords(x=origin[0], y=origin[1])
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
        if not self.map.mode == "L":
            logger.warn(
                f"Map image format is '{self.map.mode}', should be 'L' "
                "(8-bit grayscale, no alpha), converting it to 'L'."
            )
            self.map = self.map.convert("L")

    @property
    def binary_map(self) -> Image.Image:
        """The binarized occupancy map"""
        if self.negate:
            return self.map.point(lambda p: p > self.occupied_thresh and 255)
        else:
            return self.map.point(
                lambda p: 255 - p > self.occupied_thresh and 255
            )

    def pixel_from_XY(self, coords: XYCoords) -> TDRCCoords:
        """Returns the pixel index corresponding to `coods`"""
        w, h = self.map.size
        row, column = TDRC_from_XY(coords, self.origin, self.resolution, h)
        if row < 0 or row >= h or column < 0 or column >= w:
            raise ValueError(f"The point {coords} is outside the map.")
        return TDRCCoords(row, column)

    def XY_from_pixel(self, pixel: TDRCCoords) -> XYCoords:
        """Returns the x-y coords of `pixel`"""
        w, h = self.map.size
        if (
            pixel.row < 0
            or pixel.row >= h
            or pixel.column < 0
            or pixel.column >= w
        ):
            raise ValueError(f"The pixel {pixel} is outside the map.")
        return XY_from_TDRC(pixel, self.origin, self.resolution, h)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "OccupancyMap":
        """Construct `OccupancyMap` from the given metadata"""
        return cls(
            image_file=metadata["image"],
            resolution=metadata["resolution"],
            origin=metadata["origin"],
            negate=metadata["negate"],
            occupied_thresh=metadata["occupied_thresh"],
            free_thresh=metadata["free_thresh"],
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "OccupancyMap":
        """Construct `OccupancyMap` from a ROS map_server yaml metadata file"""
        with open(yaml_path, "r") as stream:
            meta = yaml.safe_load(stream)
        if not Path(meta["image"]).is_absolute():
            meta["image"] = (Path(yaml_path).parent / meta["image"]).as_posix()
        return cls.from_metadata(meta)
