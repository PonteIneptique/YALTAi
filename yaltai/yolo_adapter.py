from typing import List, Dict
from torch import hub


def segment(model: str, device: str, input: str) -> Dict[str, List[List[int]]]:
    """

    Returns {
        cls_name: [
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        ]
    }
    """
    model = hub.load("ultralytics/yolov5:v6.2", "custom", path=model, device=device)
    prediction = model(input)
    names: List[str] = list(prediction.names.values())
    out = {
        name: []
        for name in names
    }
    for i, (im, pred) in enumerate(zip(prediction.imgs, prediction.pred)):
        # img_width, img_height = im.shape[:2]

        if not pred.shape[0]:
            return {}

        for *box, conf, cls in reversed(pred):
            cls_name = names[int(cls)]
            box = [int(z.item()) for z in box]
            x0, y0, x1, y1 = box

            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            out[cls_name].append(points)

    return out
