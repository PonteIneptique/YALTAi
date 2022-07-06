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
    for i, (im, pred) in enumerate(zip(prediction.imgs, prediction.pred)):
        # img_width, img_height = im.shape[:2]

        if not pred.shape[0]:
            return {}

        for *box, conf, cls in reversed(pred):
            cls_name = names[int(cls)]
            # Top Left
            #xc = int(xc.item() * img_width)
            #yc = int(yc.item() * img_height)
            # Bottom Right
            #w = int(w.item() * img_width)
            #h = int(h.item() * img_height)
            box = [int(z.item()) for z in box]
            x0, y0, x1, y1 = box

            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            out[cls_name].append(points)

    return out
