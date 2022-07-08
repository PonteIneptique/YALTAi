""" This CLI provides tool to transform ALTO or PAGE to YOLOv5 Formats

"""
import glob
import shutil
import os
import random
import re
from typing import List, Optional
from collections import Counter

import numpy as np
from tqdm import tqdm
from PIL import Image
import click
import yaml
from kraken.kraken import message
from kraken.lib.xml import parse_xml

from yaltai.converter import AltoToYoloZone, YoloV5Zone, parse_box_labels
from yaltai.map_calc import calculate_map
from mean_average_precision import MetricBuilder


@click.group()
def cli():
    """ `yaltai` commands provides conversion options """


@cli.command("convert")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=-1)
@click.argument("output", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--segmonto", type=click.Choice(["region", "subtype", "full"]), default=None,
              help="If you use Segmonto, helper to cut the class and merge them at different levels")
@click.option("--shuffle", type=float, default=None,
              help="Split into train and val")
def convert(input: List[click.Path], output: click.Path, segmonto: Optional[str], shuffle: Optional[float]):

    if shuffle:
        message(f"Shuffling data with a ratio of {shuffle} for validation.", fg='green')
        os.makedirs(f"{output}/train/labels", exist_ok=True)
        os.makedirs(f"{output}/train/images", exist_ok=True)
        os.makedirs(f"{output}/val/labels", exist_ok=True)
        os.makedirs(f"{output}/val/images", exist_ok=True)
    else:
        os.makedirs(f"{output}/labels", exist_ok=True)
        os.makedirs(f"{output}/images", exist_ok=True)

    val_idx: Optional[int] = None
    if shuffle:
        input = list(input)
        random.shuffle(input)
        val_idx = int(len(input) * shuffle)
        message(f"{val_idx+1}/{len(input)} image for validation.", fg='green')

    def map_zones(zone_type: str) -> str:
        if segmonto:
            if segmonto == "full":
                return zone_type
            elif segmonto == "region":
                return re.search(r"([^:#]+)", zone_type).group()
            elif segmonto == "subtype":
                return re.search(r"([^#]+)", zone_type).group()
        return zone_type

    Zones: List[str] = []

    ZoneCounter = Counter()

    for idx, file in tqdm(enumerate(input)):
        parsed = parse_xml(file)
        image_path = parsed["image"]
        regions = parsed["regions"]
        for region in regions:
            if map_zones(region) not in Zones:
                Zones.append(region)

        # Retrieve image
        image = Image.open(image_path)
        width, height = image.width, image.height
        image.close()

        local_file: List[AltoToYoloZone] = []
        for region, examples in regions.items():
            region_id = Zones.index(map_zones(region))
            for box in examples:
                local_file.append(
                    AltoToYoloZone(
                        BOX=box,
                        PAGE_WIDTH=width,
                        PAGE_HEIGHT=height,
                        tag=region_id
                    )
                )
                ZoneCounter[Zones[region_id]] += 1

        path = output
        if shuffle:
            path = f"{output}/train"
            if idx <= val_idx:
                path = f"{output}/val"

        src_img = image_path
        ext = src_img.split(".")[-1]
        simplified_name = '.'.join(os.path.basename(image_path).split('.')[:-1])

        if ext.lower() not in {"jpg", "jpeg"}:
            # open image in png format
            img_png = Image.open(src_img)

            # The image object is used to save the image in jpg format
            img_png.save(f"{path}/images/{simplified_name}.jpg")
            img_png.close()
        else:
            shutil.copy(src_img, f"{path}/images/{simplified_name}.jpg")

        with open(f"{path}/labels/{simplified_name}.txt", "w") as f:
            f.write("\n".join([loc.yoloV5() for loc in local_file]))

    message(f"{len(input)} ground truth XML files converted.", fg='green')

    with open(f"{output}/config.yml", "w") as f:
        data = {
            "train": os.path.abspath(output),
            "val": os.path.abspath(output),
            "nc": len(Zones),
            "names": Zones
        }
        if shuffle:
            data.update({
                "train": f"{os.path.abspath(output)}/train/images",
                "val": f"{os.path.abspath(output)}/val/images"
            })

        yaml.dump(
            data=data,
            stream=f,
            sort_keys=False
        )

    with open(f"{output}/labelmap.txt", "w") as f:
        f.write("\n".join(Zones))

    message(f"Configuration available at {output}/config.yml.", fg='green')
    message(f"Label Map available at {output}/labelmap.txt.", fg='green')

    message(f"Regions count:", fg='blue')
    for zone, cnt in ZoneCounter.items():
        message(f"\t- {cnt:05} {zone}", fg='blue')


@cli.command("scores")
@click.argument("gt-directory", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("pred-directory", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option("-t", "--threshold", type=float, help="IoU Threshold", default=.5, show_default=True)
def get_scores(gt_directory, pred_directory, threshold):
    gt_directory = os.path.join(gt_directory, "*.txt")
    pred_directory = os.path.join(pred_directory, "*.txt")

    ground_truth, gt_arrays = parse_box_labels(sorted(glob.glob(gt_directory)))
    pred, pred_arrays = parse_box_labels(sorted(glob.glob(pred_directory)), gt=False)

    classes = np.unique(
        np.concatenate((
            np.array([row for arr in gt_arrays for row in arr])[:, 4],
            np.array([row for arr in pred_arrays for row in arr])[:, 4]
        ))
    ).tolist()

    def reclass_classes(array_list: List[np.array]) -> None:
        for array in array_list:
            for row_idx in range(array.shape[0]):
                if array[row_idx, 4].astype(int) > 9:
                    print(array[row_idx, 4].astype(int), array[row_idx], classes)
                array[row_idx, 4] = classes.index(array[row_idx, 4].astype(int))

    reclass_classes(gt_arrays)
    reclass_classes(pred_arrays)

    builder = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=len(classes))
    for pred_array, gt_array in zip(pred_arrays, gt_arrays):
        builder.add(pred_array, gt_array)

    metric = builder.value(iou_thresholds=0.5)
    print(f"VOC PASCAL mAP: {metric['mAP']}")
    for cls_idx, cls_orig_idx in enumerate(classes):
        data = metric[0.5][cls_idx]
        ap, precision, recall, support = data["ap"], data["precision"].mean(), data["recall"].mean(), \
                                         data["recall"].shape[0]
        print(f"AP={ap}, PRE={precision}, REC={recall}, SUP={support}")


if __name__ == "__main__":
    cli()
