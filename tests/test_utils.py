import os
import numpy as np
from yaltai.utils import read_labelmap, parse_box_labels, XYXY


def test_read_labelmap():
    """Asserts that reading a label map works"""
    labels = read_labelmap(os.path.join(
        os.path.dirname(__file__),
        "test_files",
        "label_map.txt"
    ))
    assert labels == ["Class0", "Class1", "Stuff"]


def test_read_files():
    """Asserts that parsing COCO/YOLO formats work"""
    annots, arrays = parse_box_labels([
        os.path.join(
            os.path.dirname(__file__),
            "test_files",
            "annot1.txt"
        ),
        os.path.join(
            os.path.dirname(__file__),
            "test_files",
            "annot2.txt"
        )
    ])
    assert annots == {
        'boxes': [
            XYXY(x0=15, y0=33, x1=82, y1=41),
            XYXY(x0=15, y0=15, x1=83, y1=31),
            XYXY(x0=35, y0=11, x1=62, y1=14),
            XYXY(x0=79, y0=12, x1=82, y1=15),
            XYXY(x0=16, y0=43, x1=83, y1=74),
            XYXY(x0=16, y0=77, x1=85, y1=88),
            XYXY(x0=84, y0=37, x1=99, y1=56),
            XYXY(x0=16, y0=25, x1=81, y1=66),
            XYXY(x0=17, y0=66, x1=81, y1=89),
            XYXY(x0=77, y0=10, x1=80, y1=13),
            XYXY(x0=26, y0=10, x1=70, y1=13),
            XYXY(x0=16, y0=15, x1=80, y1=25),
            XYXY(x0=35, y0=13, x1=60, y1=15)
        ],
        'labels': [14, 14, 35, 31, 14, 14, 1, 23, 23, 31, 35, 24, 28]
    }
    assert (arrays[0] == np.array([[15, 33, 82, 41, 14, 0, 0],
                      [15, 15, 83, 31, 14, 0, 0],
                      [35, 11, 62, 14, 35, 0, 0],
                      [79, 12, 82, 15, 31, 0, 0],
                      [16, 43, 83, 74, 14, 0, 0],
                      [16, 77, 85, 88, 14, 0, 0],
                      [84, 37, 99, 56, 1, 0, 0]])).all()
    assert (arrays[1] == np.array([[16, 25, 81, 66, 23, 0, 0],
                      [17, 66, 81, 89, 23, 0, 0],
                      [77, 10, 80, 13, 31, 0, 0],
                      [26, 10, 70, 13, 35, 0, 0],
                      [16, 15, 80, 25, 24, 0, 0],
                      [35, 13, 60, 15, 28, 0, 0]])).all()
