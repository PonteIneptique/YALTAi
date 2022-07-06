from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class YoloZone:
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
