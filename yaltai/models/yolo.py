from typing import List, Dict
from ultralytics import YOLO
from ultralytics.engine.results import Results

from yaltai.preprocessing import deskew, rotatebox


def segment(
        model: YOLO,
        input: str,
        apply_deskew: bool = False,
        max_angle: float = 10.0
) -> Dict[str, List[List[int]]]:
    """

    Returns {
        cls_name: [
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        ]
    }
    """
    rotated_input = None
    angle = 0
    predictions: List[Results] = []
    if apply_deskew:
        rotated_input, angle = deskew(input)
        if abs(angle) > max_angle:
            predictions = model.predict(input, save=False)
            rotated_input = None
        else:
            predictions = model.predict(rotated_input, save=False)
    else:
        predictions = model.predict(input, save=False)

    names: List[str] = list(set([
        name
        for res in predictions
        for name in res.names.values()
    ]))

    out = {
        name: []
        for name in names
    }
    for pred in predictions:
        for box, cls_id in zip(pred.boxes.xyxy, pred.boxes.cls):
            cls_name = pred.names[cls_id.item()]
            x0, y0, x1, y1 = box.tolist()
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]

            if apply_deskew and rotated_input is not None:
                points = rotatebox(points, rotated_input, -angle)
                points.append(points[0])
            out[cls_name].append(points)

    return out
