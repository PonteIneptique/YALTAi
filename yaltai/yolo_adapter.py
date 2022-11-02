from typing import List, Dict
from torch import hub

from yaltai.preprocessing import deskew, rotatebox


def segment(
        model: str,
        device: str,
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
    model = hub.load("ultralytics/yolov5:v6.2", "custom", path=model, device=device)
    model.eval()
    rotated_input = None
    angle = 0
    if apply_deskew:
        rotated_input, angle = deskew(input)
        if abs(angle) > max_angle:
            prediction = model(input)
            rotated_input = None
        else:
            prediction = model(rotated_input)
    else:
        prediction = model(input)

    if isinstance(prediction.names, dict):
        names: List[str] = list(prediction.names.values())
    else:
        names: List[str] = list(prediction.names)

    out = {
        name: []
        for name in names
    }
    for i, (im, pred) in enumerate(zip(prediction.imgs, prediction.pred)):
        if not pred.shape[0]:
            return {}
        for *box, conf, cls in reversed(pred):
            cls_name = names[int(cls)]
            box = [int(z.item()) for z in box]
            x0, y0, x1, y1 = box

            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            if apply_deskew and rotated_input is not None:
                points = rotatebox(points, rotated_input, -angle)
                points.append(points[0])
            out[cls_name].append(points)

    return out
#https://traces6.paris.inria.fr/document/2084/part/220680/edit/
#yaltai kraken -i ../valais-data/batch-14-FR/AEV_3090_1880_Monthey_Collombey-Muraz_Collombey_020.jpg ../valais-data/batch-14-FR/AEV_3090_1880_Monthey_Collombey-Muraz_Collombey_020.xml -f image --raise-on-error segment -y ../valais-recensement/yolov5/runs/train/exp7/weights/best.pt