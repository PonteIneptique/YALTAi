import click
import os
import dataclasses
from typing import cast
from kraken.kraken import (
    # Constants
    SEGMENTATION_DEFAULT_MODEL,
    # CLI Stuff
    message, logger,  # Logics
    get_input_parser, partial
)
from PIL import Image
from kraken.containers import Segmentation
from ultralytics import YOLO


def segmenter(model, text_direction, mask, device, yolo_model, ignore_lines, deskew, max_angle, input, output) -> None:
    import json
    import yaltai.models.krakn
    import yaltai.models.yolo

    ctx = click.get_current_context()

    if ctx.meta['first_process']:
        if ctx.meta['input_format_type'] != 'image':
            input = get_input_parser(ctx.meta['input_format_type'])(input).imagename
        ctx.meta['first_process'] = False

    if 'base_image' not in ctx.meta:
        ctx.meta['base_image'] = input

    try:
        im = Image.open(input)
    except IOError as e:
        raise click.BadParameter(str(e))

    if mask:
        try:
            mask = Image.open(mask)
        except IOError as e:
            raise click.BadParameter(str(e))

    message(f'Segmenting {ctx.meta["orig_file"]}\t', nl=False)
    try:
        regions = yaltai.models.yolo.segment(
            yolo_model, input=input,
            apply_deskew=deskew, max_angle=max_angle
        )
        res: Segmentation = yaltai.models.krakn.segment(
            im, text_direction, mask=mask, model=model, device=device,
            regions=regions, ignore_lignes=ignore_lines,
            raise_on_error=ctx.meta['raise_failed'], autocast=ctx.meta["autocast"]
        )
    except Exception as E:
        if ctx.meta['raise_failed']:
            raise
        message('\u2717', fg='red')
        ctx.exit(1)

    if ctx.meta['last_process'] and ctx.meta['output_mode'] != 'native':
        with click.open_file(output, 'w', encoding='utf-8') as fp:
            fp = cast('IO[Any]', fp)
            logger.info('Serializing as {} into {}'.format(ctx.meta['output_mode'], output))
            from kraken import serialization
            fp.write(
                serialization.serialize(
                    results=res,
                    image_size=im.size,
                    template=ctx.meta['output_template'],
                    template_source='custom' if ctx.meta['output_mode'] == 'template' else 'native',
                    processing_steps=ctx.meta['steps']
                )
            )
    else:
        with click.open_file(output, 'w') as fp:
            fp = cast('IO[Any]', fp)
            json.dump(dataclasses.asdict(res), fp)
    message('\u2713', fg='green')


from kraken.kraken import cli as kcli


@kcli.command('segment')
@click.pass_context
@click.option('-i', '--model',
              default=None,
              show_default=True, type=click.Path(exists=True),
              help='Baseline detection model to use')
@click.option('-y', '--yolo',
              default=None,
              show_default=True, type=click.Path(exists=True),
              help='YOLO model to use')
@click.option('-d', '--text-direction', default='horizontal-lr',
              show_default=True,
              type=click.Choice(['horizontal-lr', 'horizontal-rl',
                                 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction')
@click.option('-m', '--mask', show_default=True, default=None,
              type=click.File(mode='rb', lazy=True), help='Segmentation mask '
              'suppressing page areas for line detection. 0-valued image '
              'regions are ignored for segmentation purposes. Disables column '
              'detection.')
@click.option('-d', '--deskew', show_default=True, default=False, is_flag=True,
              help='Prior to applying YOLO model, '
                   'deskew the image: this will produced oriented bounding box. The final output'
                   'is realigned with the original image.')
@click.option('--max-angle', show_default=True, default=10, type=float,
              help='Maximum deskewing angle')
@click.option('-n', '--ignore-lines', show_default=True, default=False, is_flag=True,
              help='Does not run line segmentation through Kraken, only Zone from YOLO')
def yaltai_segment(ctx, model, text_direction, mask, yolo, ignore_lines, deskew, max_angle):
    """
    Segments page images into text lines.
    """

    if not model:
        model = SEGMENTATION_DEFAULT_MODEL
    if not yolo:
        raise Exception("No YOLOv8 model given")
    ctx.meta['steps'].append({'category': 'processing',
                              'description': 'Baseline and region segmentation',
                              'settings': {'model': os.path.basename(model),
                                           'text_direction': text_direction}})

    from kraken.lib.vgsl import TorchVGSLModel
    message(f'Loading ANN {model}\t', nl=False)
    try:
        model = TorchVGSLModel.load_model(model)
        model.to(ctx.meta['device'])
    except Exception:
        if ctx.meta['raise_failed']:
            raise
        message('\u2717', fg='red')
        ctx.exit(1)

        message('\u2713', fg='green')

    yolo = YOLO(yolo)
    yolo.to(ctx.meta["device"])

    return partial(segmenter, model, text_direction, mask, ctx.meta['device'], yolo, ignore_lines, deskew, max_angle)
