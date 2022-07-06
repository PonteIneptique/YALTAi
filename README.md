# YALTAi
You Actually Look Twice At it

This provides an adapter for Kraken to use YOLOv5 Object Detection routine.

This tool can be used for both segmenting and conversion of models.

# Routine

```bash
pip install YALTAi
# Keeps .1 data in the validation set and convert all alto into alto
#  Keeps the segmonto information up to the regions
python -m yaltai.yaltai convert PATH/TO/ALTOorPAGE/*.xml my-dataset --shuffle .1 --segmonto region
# Train your YOLOv5 data (YOLOv5 is installed with YALTAi)
yolov5 train --data "$PWD/my-dataset/config.yml" --batch-size 4 --img 640 --weights yolov5x.pt --epochs 50
# Retrieve the best.pt after the training
# It should be in runs/train/exp[NUMBER]/weights/best.pt
# And then annotate your new data with the same CLI API as Kraken !
python -m yaltai.kraken_yaltai --device cuda:0 -I "*.jpg" --suffix ".xml" segment --yolo runs/train/exp5/weights/best.pt
```