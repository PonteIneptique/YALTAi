import click
import os
import dataclasses
from typing import Dict, IO, Any, cast
from kraken.kraken import (
    # Constants
    SEGMENTATION_DEFAULT_MODEL,
    # CLI Stuff
    message, logger, log,
    # Logics
    get_input_parser, partial
)
from PIL import Image
from kraken.lib.progress import KrakenProgressBar
from kraken.containers import Segmentation
from ultralytics import YOLO


def segmenter(model, text_direction, mask, device, yolo_model, ignore_lines, deskew, max_angle, input, output) -> None:
    import json
    import yaltai.kraken_adapter
    import yaltai.yolo_adapter

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
        regions = yaltai.yolo_adapter.segment(
            yolo_model, input=input,
            apply_deskew=deskew, max_angle=max_angle
        )
        res: Segmentation = yaltai.kraken_adapter.segment(
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

#
# @click.group(chain=True)
# @click.version_option()
# @click.option('-i', '--input',
#               type=(click.Path(exists=True),  # type: ignore
#                     click.Path(writable=True)),
#               multiple=True,
#               help='Input-output file pairs. Each input file (first argument) is mapped to one '
#                    'output file (second argument), e.g. `-i input.png output.txt`')
# @click.option('-f', '--format-type', type=click.Choice(['image', 'pdf']), default='image',
#               help='Sets the default input type. In image mode inputs are image '
#                    'files, pdf '
#                    'expects PDF files with numbered suffixes added to output file '
#                    'names as needed.')
# @click.option('-I', '--batch-input', multiple=True, help='Glob expression to add multiple files at once.')
# @click.option('-r', '--raise-on-error/--no-raise-on-error', default=False, show_default=True,
#               help='Raises the exception that caused processing to fail in the case of an error')
# @click.option('-v', '--verbose', default=0, count=True, show_default=True)
# @click.option('-d', '--device', default='cpu', show_default=True,
#               help='Select device to use (cpu, cuda:0, cuda:1, ...)')
# @click.option('-o', '--suffix', default='', show_default=True,
#               help='Suffix for output files from batch and PDF inputs.')
# @click.option('-p', '--pdf-format', default='{src}_{idx:06d}',
#               show_default=True,
#               help='Format for output of PDF files. valid fields '
#                    'are `src` (source file), `idx` (page number), and `uuid` (v4 uuid). '
#                    '`-o` suffixes are appended to this format string.')
# def cli(device, input, batch_input, raise_on_error, format_type, verbose, suffix, pdf_format):
#     """ YALTAi is built as a group of command but only takes one command at the time: segment """
#     ctx = click.get_current_context()
#     if device != 'cpu':
#         import torch
#         try:
#             torch.ones(1, device=device)
#         except AssertionError as e:
#             if raise_on_error:
#                 raise
#             logger.error(f'Device {device} not available: {e.args[0]}.')
#             ctx.exit(1)
#     ctx.meta['device'] = device
#     ctx.meta['input_format_type'] = format_type if format_type != 'pdf' else 'image'
#     ctx.meta['raise_failed'] = raise_on_error
#     ctx.meta['output_mode'] = "alto"  # Unlike Kraken, forces ALTO
#     ctx.meta['verbose'] = verbose
#     ctx.meta['steps'] = []
#     log.set_logger(logger, level=30 - min(10 * verbose, 20))


from kraken.kraken import cli


@cli.command('segment')
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
        raise Exception("No YOLOv5 model given")
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


if __name__ == "__main__":
    cli()
