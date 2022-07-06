""" This CLI provides tool to transform ALTO or PAGE to YOLOv5 Formats

"""


import shutil
import os
import random
import re
from typing import List, Optional
from collections import Counter

from tqdm import tqdm
from PIL import Image
import click
import yaml
from kraken.kraken import message

from yaltai.converter import YoloZone
from kraken.lib.xml import parse_xml


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
        image_path = os.path.join(os.path.dirname(file), parsed["image"])
        regions = parsed["regions"]
        for region in regions:
            if map_zones(region) not in Zones:
                Zones.append(region)

        # Retrieve image
        image = Image.open(image_path)
        width, height = image.width, image.height
        image.close()

        local_file: List[YoloZone] = []
        for region, examples in regions.items():
            region_id = Zones.index(map_zones(region))
            for box in examples:
                local_file.append(
                    YoloZone(
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
            "train": output,
            "val": output,
            "nc": len(Zones),
            "names": Zones
        }
        if shuffle:
            data.update({
                "train": f"{output}/train/images",
                "val": f"{output}/val/images"
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


if __name__ == "__main__":
    cli()
