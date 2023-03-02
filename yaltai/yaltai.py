""" This CLI provides tool to transform ALTO or PAGE to YOLOv5 Formats

"""
import glob
import json
import shutil
import os
import random
import re
import sys
from typing import List, Optional
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
import click
import yaml
from kraken.kraken import message
from kraken.lib.xml import parse_xml
import tabulate

from yaltai.converter import AltoToYoloZone, parse_box_labels, read_labelmap, YoloV5Zone
from mean_average_precision import MetricBuilder


@click.group()
def cli():
    """ `yaltai` commands provides conversion options """


@cli.command("alto-to-yolo")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=-1)
@click.argument("output", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--segmonto", type=click.Choice(["region", "subtype", "full"]), default=None,
              help="If you use Segmonto, helper to cut the class and merge them at different levels")
@click.option("--shuffle", type=float, default=None,
              help="Split into train and val")
@click.option("-l", "--labelmap", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Format for the score table", default=None, show_default=True)
@click.option("--image/--no-image", type=bool, default=True, show_default=True,
              help="Copy images when converting ALTO to YOLOv5")
def convert(input: List[click.Path], output: click.Path, segmonto: Optional[str], shuffle: Optional[float],
            labelmap: Optional[str], image: bool):
    """ Converts ALTO-XML files to YOLOv5 training files
    """
    if shuffle:
        message(f"Shuffling data with a ratio of {shuffle} for validation.", fg='green')
        os.makedirs(f"{output}/train/labels", exist_ok=True)
        os.makedirs(f"{output}/val/labels", exist_ok=True)
        if image:
            os.makedirs(f"{output}/train/images", exist_ok=True)
            os.makedirs(f"{output}/val/images", exist_ok=True)
    else:
        os.makedirs(f"{output}/labels", exist_ok=True)
        if image:
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
    if labelmap:
        Zones = read_labelmap(labelmap)

    ZoneCounter = Counter()

    for idx, file in tqdm(enumerate(input)):
        parsed = parse_xml(file)
        image_path: Path = parsed["image"]
        regions = parsed["regions"]
        for region in regions:
            if map_zones(region) not in Zones:
                Zones.append(map_zones(region))

        # Retrieve image
        image_file = Image.open(image_path)
        width, height = image_file.width, image_file.height
        image_file.close()

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
        ext = src_img.suffix[1:]  # Suffix keeps the dot, we remove it
        simplified_name = src_img.stem

        if image:
            if ext.lower() not in {"jpg", "jpeg"}:
                # open image in png format
                img_png = Image.open(src_img)

                if img_png.mode == "RGBA":  # Handle RGBA
                    img_png = img_png.convert('RGB')

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
@click.option("-f", "--format", type=click.Choice(["markdown", "latex"]),
              help="Format for the score table", default="markdown", show_default=True)
@click.option("-l", "--labelmap", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Labelmap to print nicely the information", default=None, show_default=True)
@click.option("-j", "--save-json", type=click.File(mode="w"),
              help="JSON File to save information", default=None, show_default=True)
def get_scores(gt_directory, pred_directory, threshold, format, labelmap, save_json):
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
                array[row_idx, 4] = classes.index(array[row_idx, 4].astype(int))

    reclass_classes(gt_arrays)
    reclass_classes(pred_arrays)

    builder = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=len(classes))
    for pred_array, gt_array in zip(pred_arrays, gt_arrays):
        builder.add(pred_array, gt_array)

    metric = builder.value(iou_thresholds=threshold)
    print(f"Global mAP: {metric['mAP']}")

    if labelmap:
        labelmap = read_labelmap(labelmap)
    else:
        labelmap = list(range(max(classes) + 1))

    table = [["Class", "AP", "Precision", "Recall", "Support"]]
    for cls_idx, cls_orig_idx in enumerate(classes):
        data = metric[0.5][cls_idx]
        ap, precision, recall, support = data["ap"], data["precision"].mean(), data["recall"].mean(), \
                                         data["precision"].shape[0]
        table.append([labelmap[cls_orig_idx], ap, precision, recall, support])

    print(tabulate.tabulate(table, tablefmt=format, floatfmt=".3f", headers="firstrow"))

    if save_json is not None:
        json.dump({
            "mAP": float(metric["mAP"]),
            "classes": {
                row[0]: {
                    "AP": float(row[1]),
                    "Precision": float(row[2]),
                    "Recall": float(row[2]),
                    "Support": float(row[3])
                }
                for row in table[1:]
            }
        }, save_json)


@cli.command("yolo-to-alto")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=-1)
@click.option("-l", "--labelmap", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Format for the score table", default=None, show_default=True)
def yolo_to_alto(input, labelmap):
    """ Converts YOLOv5.txt files to ALTO files """
    if not labelmap:
        message("No labelmap given, --labelmap is required for ALTO conversion", fg="red")
        sys.exit(0)

    labelmap = read_labelmap(labelmap)

    OtherTags = "\n".join([
        f'<OtherTag ID="BT{idx:03}" LABEL="{zone}" DESCRIPTION="block type {zone}"/>'
        for idx, zone in enumerate(labelmap)
    ])

    with open(os.path.join(os.path.dirname(__file__), "template.xml")) as f:
        TEMPLATE = f.read()

    for file in input:
        xml_name = file[:-4] + ".xml"
        img_file_name = os.path.basename(file[:-4]) + ".jpg"
        zones = []
        if os.path.exists(os.path.join(os.path.dirname(file), "..", "images", img_file_name)):
            img_name = os.path.join(os.path.dirname(file), "..", "images", img_file_name)
            img_for_xml_name = f"../images/{img_file_name}"
        elif os.path.exists(os.path.join(os.path.dirname(file), "..", "images", img_file_name)):
            img_name = os.path.join(os.path.dirname(file), "..", "images", img_file_name)
            img_for_xml_name = os.path.join("..", img_file_name)
        else:
            message(f"Can't find the image for {img_file_name}")
            sys.exit(0)

        image = Image.open(img_name)
        img_width, img_height = image.size
        image.close()

        with open(file) as f:
            for line_idx, line in enumerate(f):
                z = YoloV5Zone.from_txt(*line.strip().split()[:5])

                x0, y0, x1, y1 = z.xyxy
                x0, x1 = img_width*x0, img_width*x1
                y0, y1 = img_height*y0, img_height*y1
                x0, x1, y0, y1 = [int(z) for z in [x0, x1, y0, y1]]
                width = x1 - x0
                height = y1 - y0

                zones.append(f"""
            <TextBlock HPOS="{x0}" VPOS="{y0}"
                       WIDTH="{int(width)}" HEIGHT="{int(height)}"
                       ID="eSc_textblock_blck{line_idx:03}"
                       TAGREFS="BT{z.tag:03}">
                <Shape>
                    <Polygon POINTS="{x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1} {x0} {y0}"/>
                </Shape>
            </TextBlock>""")

        with open(xml_name, "w") as f:
            f.write(
                TEMPLATE.replace("%Filename%", img_for_xml_name)
                .replace("%Width%", str(img_width))
                .replace("%Height%", str(img_height))
                .replace("%Tags%", OtherTags)
                .replace("%Textblocks%", "".join(zones))
            )


import yaltai.kraken_yaltai as kyaltai

cli.add_command(kyaltai.cli, "kraken")


if __name__ == "__main__":
    cli()
