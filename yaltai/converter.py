from dataclasses import dataclass
from collections import namedtuple
from typing import List, Tuple, Optional, Dict, Union
import numpy as np

XYXY = namedtuple("XYXY", ["x0", "y0", "x1", "y1"])


@dataclass
class AltoToYoloZone:
    BOX: List[Tuple[int, int]]
    PAGE_WIDTH: int
    PAGE_HEIGHT: int
    tag: int
    _xywh: Optional[Tuple[int, int, int, int]] = None

    @property
    def height(self):
        return self.xywh[-1]

    @property
    def width(self):
        return self.xywh[-2]

    @property
    def x_center(self) -> int:
        return int(self.width / 2 + self.xywh[0])

    @property
    def y_center(self) -> int:
        return int(self.height / 2 + self.xywh[1])

    @property
    def xywh(self):
        if self._xywh:
            return self._xywh

        box = np.array(self.BOX)
        x_min, y_min = box.min(axis=0)
        x_max, y_max = box.max(axis=0)

        width = x_max - x_min
        height = y_max - y_min

        self._xywh = (x_min, y_min, width, height)
        return self._xywh

    def yoloV5(self):
        return (f"{self.tag}"
                f" {self.x_center / self.PAGE_WIDTH:.6f}"
                f" {self.y_center / self.PAGE_HEIGHT:.6f}"
                f" {self.width / self.PAGE_WIDTH:.6f}"
                f" {self.height / self.PAGE_HEIGHT:.6f}")


@dataclass
class YoloV5Zone:
    tag: int
    xc: float
    yc: float
    w: float
    h: float

    @classmethod
    def from_txt(cls, tag, *box):
        return YoloV5Zone(int(tag), *[float(b) for b in box])

    @property
    def xyxy(self):
        return XYXY(
            self.xc - self.w / 2,
            self.yc - self.h / 2,
            self.xc + self.w / 2,
            self.yc + self.h / 2
        )

    @property
    def xyxy100(self):
        return XYXY(
            *[int(100 * b) for b in self.xyxy]
        )


def parse_box_labels(
    files: List[str],
    gt: bool = True
) -> Tuple[Dict[str, Union[List[Union[XYXY, int]]]], List[np.array]]:
    parsed = {"boxes": [], "labels": []}
    arrays = []
    for file in sorted(files):
        with open(file) as f:
            start_index = len(parsed["boxes"])
            for line in f:
                z = YoloV5Zone.from_txt(*line.strip().split()[:5])
                parsed["boxes"].append(z.xyxy100)
                parsed["labels"].append(z.tag)
            arrays.append(
                np.array([
                    [*xyxy, cls_idx] + ([0, 0] if gt else [1])
                    for xyxy, cls_idx in zip(parsed["boxes"][start_index:], parsed["labels"][start_index:])
                ])
            )

    return parsed, arrays


def read_labelmap(path: str) -> List[str]:
    lines = []
    with open(path) as f:
        lines = f.read().split()
    return lines
