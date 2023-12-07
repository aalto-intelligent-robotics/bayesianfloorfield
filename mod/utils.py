import json
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from jsonschema import Draft7Validator, exceptions, validators
from pydantic import NonNegativeInt

MIN_DISTANCE = 0.000001

GROUP_DISTANCE_TOLERANCE = 0.1


_2PI = np.pi * 2


class XYCoords(NamedTuple):
    x: float
    y: float


class RCCoords(NamedTuple):
    row: NonNegativeInt
    column: NonNegativeInt


def RC_from_XY(xy: XYCoords, origin: XYCoords, resolution: float) -> RCCoords:
    row = int((xy.y - origin.y) // resolution)
    col = int((xy.x - origin.x) // resolution)
    return RCCoords(row, col)


def XY_from_RC(rc: RCCoords, origin: XYCoords, resolution: float) -> XYCoords:
    x = rc.column * resolution + origin.x
    y = rc.row * resolution + origin.y
    return XYCoords(x, y)


class Direction(IntEnum):
    E = 0
    NE = 1
    N = 2
    NW = 3
    W = 4
    SW = 5
    S = 6
    SE = 7

    @property
    def rad(self) -> float:
        return self.value * _2PI / 8

    @property
    def range(self) -> tuple[float, float]:
        a = self.rad
        return ((a - np.pi / 8) % _2PI, a + np.pi / 8)

    @property
    def u(self) -> float:
        return np.cos(self.rad)

    @property
    def v(self) -> float:
        return np.sin(self.rad)

    @property
    def uv(self) -> tuple[float, float]:
        return (self.u, self.v)

    def contains(self, rad: float) -> bool:
        a = rad % _2PI
        s, e = self.range
        return (a - s) % _2PI < (e - s) % _2PI

    @classmethod
    def from_rad(cls, rad: float) -> "Direction":
        for dir in Direction:
            if dir.contains(rad):
                return dir
        raise ValueError(f"{rad} cannot be represented as Direction")

    @classmethod
    def from_points(
        cls, p1: tuple[float, float], p2: tuple[float, float]
    ) -> "Direction":
        assert p1 != p2
        rad = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        return cls.from_rad(rad)


def extended_validator(json_path, schema_path):
    schema_file = open(schema_path, "r")
    my_schema = json.load(schema_file)

    json_file = open(json_path, "r")
    my_json = json.load(json_file)

    validate_properties = Draft7Validator.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, sub_schema in properties.items():
            if "default" in sub_schema:
                instance.setdefault(prop, sub_schema["default"])

        for error in validate_properties(
            validator,
            properties,
            instance,
            schema,
        ):
            yield error

    ext_validator = validators.extend(
        Draft7Validator,
        {"properties": set_defaults},
    )

    try:
        ext_validator(my_schema).validate(my_json)
    except exceptions.ValidationError as e:
        return False, e
    except exceptions.SchemaError as e:
        return False, e
    return True, my_json


def get_local_settings(
    json_path="config/local_settings.json",
    schema_path="config/local_settings_schema.json",
):
    return extended_validator(json_path, schema_path)
