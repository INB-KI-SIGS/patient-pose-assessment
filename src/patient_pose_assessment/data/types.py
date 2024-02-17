from typing import Literal, List

SIDE_TYPE = Literal["r", "l"]
SIDES: List[SIDE_TYPE] = [
    "l",
    "r",
]

CAMERA_TYPE = Literal["cam_left", "cam_right"]
CAMERAS: List[CAMERA_TYPE] = [
    "cam_left",
    "cam_right",
]

FLEXION_TYPE = Literal["max", "medium", "min"]
FLEXIONS: List[FLEXION_TYPE] = ["max", "medium", "min"]
