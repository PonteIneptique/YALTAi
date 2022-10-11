# YALTAi
You Actually Look Twice At it

This provides an adapter for Kraken to use YOLOv5 Object Detection routine.

This tool can be used for both segmenting and conversion of models.

# Routine

## Instal

```bash
pip install YALTAi
```

## Training

Convert (and split optionally) your data

```bash
# Keeps .1 data in the validation set and convert all alto into YOLOv5 format
#  Keeps the segmonto information up to the regions
yaltai alto-to-yolo PATH/TO/ALTOorPAGE/*.xml my-dataset --shuffle .1 --segmonto region
```

And then [train YOLO](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) 

```bash
# Download YOLOv5
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
git checkout v6.2
pip install -r requirements.txt  # install
# Train your YOLOv5 data (YOLOv5 is installed with YALTAi)
python train.py --data "../my-dataset/config.yml" --batch-size 4 --img 640 --weights yolov5x.pt --epochs 50
```

## Predicting

YALTAi has the same CLI interface as Kraken, so:

- You can use base BLLA model for line or provide yours with `-m model.mlmodel`
- Use a GPU (`--device cuda:0`) or a CPU (`--device cpu`)
- Apply on batch (`*.jpg`)

```bash
# Retrieve the best.pt after the training
# It should be in runs/train/exp[NUMBER]/weights/best.pt
# And then annotate your new data with the same CLI API as Kraken !
yaltai kraken --device cuda:0 -I "*.jpg" --suffix ".xml" segment --yolo runs/train/exp5/weights/best.pt
```

## Metrics

The metrics produced from various libraries never gives the same mAP or Precision. I tried

- `object-detection-metrics==0.4`
- `mapCalc`
- `mean-average-precision` which ended up being the chosen one (cleanest in terms of how I can access info) 

and of course I compared with YOLOv5 raw results. Nothing worked the same. And the library YOLOv5 derives its metrics from is uninstallable through pip.
