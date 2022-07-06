from typing import List, Dict
from yolov5 import YOLOv5
from yolov5.models.common import Detections


def segment(model: str, device: str, input: str) -> Dict[str, List[List[int]]]:
    """

    Returns {
        cls_name: [
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        ]
    }
    """
    model = YOLOv5(model_path=model, device=device)
    prediction: Detections = model.predict([input])
    names: List[str] = prediction.names
    out = {
        name: []
        for name in names
    }
    for i, (im, pred) in enumerate(zip(prediction.imgs, prediction.xyxyn)):
        img_width, img_height = im.shape[:2]

        if not pred.shape[0]:
            return {}

        for x0, y0, x1, y1, conf, cls in reversed(pred):
            cls_name = names[int(cls)]
            # Top Left
            x0 = int(float(x0) * img_width)
            x1 = int(float(x1) * img_width)
            # Bottom Right
            y0 = int(float(y0) * img_height)
            y1 = int(float(y1) * img_height)

            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            out[cls_name].append(points)

    return out
