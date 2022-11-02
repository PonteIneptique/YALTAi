import click
import os
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


def segmenter(model, text_direction, mask, device, yolo_model, ignore_lines, deskew, max_angle, input, output) -> None:
    import json
    import yaltai.kraken_adapter
    import yaltai.yolo_adapter

    ctx = click.get_current_context()

    if ctx.meta['first_process']:
        if ctx.meta['input_format_type'] != 'image':
            input = get_input_parser(ctx.meta['input_format_type'])(input)['image']
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

    message('Segmenting\t', nl=False)
    res: Dict[str, Any] = None
    try:
        regions = yaltai.yolo_adapter.segment(
            yolo_model,
            device=device, input=input,
            apply_deskew=deskew, max_angle=max_angle
        )
        res = yaltai.kraken_adapter.segment(
            im, text_direction, mask=mask, model=model, device=device,
            regions=regions, ignore_lignes=ignore_lines
        )
    except Exception as E:
        if ctx.meta['raise_failed']:
            raise
        message('\u2717', fg='red')
        ctx.exit(1)

    if ctx.meta['last_process'] and ctx.meta['output_mode'] != 'native':
        with click.open_file(output, 'w', encoding='utf-8') as fp:
            fp = cast(IO[Any], fp)
            logger.info('Serializing as {} into {}'.format(ctx.meta['output_mode'], output))
            from kraken import serialization
            fp.write(serialization.serialize_segmentation(res,
                                                          image_name=ctx.meta['base_image'],
                                                          image_size=im.size,
                                                          template=ctx.meta['output_mode'],
                                                          ))  # Todo: Add `processing_steps=ctx.meta['steps']))` when
                                                              #  new version releases
    else:
        with click.open_file(output, 'w') as fp:
            fp = cast(IO[Any], fp)
            json.dump(res, fp)
    message('\u2713', fg='green')


@click.group(chain=True)
@click.version_option()
@click.option('-i', '--input',
              type=(click.Path(exists=True),  # type: ignore
                    click.Path(writable=True)),
              multiple=True,
              help='Input-output file pairs. Each input file (first argument) is mapped to one '
                   'output file (second argument), e.g. `-i input.png output.txt`')
@click.option('-f', '--format-type', type=click.Choice(['image', 'pdf']), default='image',
              help='Sets the default input type. In image mode inputs are image '
                   'files, pdf '
                   'expects PDF files with numbered suffixes added to output file '
                   'names as needed.')
@click.option('-I', '--batch-input', multiple=True, help='Glob expression to add multiple files at once.')
@click.option('-r', '--raise-on-error/--no-raise-on-error', default=False, show_default=True,
              help='Raises the exception that caused processing to fail in the case of an error')
@click.option('-v', '--verbose', default=0, count=True, show_default=True)
@click.option('-d', '--device', default='cpu', show_default=True,
              help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('-o', '--suffix', default='', show_default=True,
              help='Suffix for output files from batch and PDF inputs.')
@click.option('-p', '--pdf-format', default='{src}_{idx:06d}',
              show_default=True,
              help='Format for output of PDF files. valid fields '
                   'are `src` (source file), `idx` (page number), and `uuid` (v4 uuid). '
                   '`-o` suffixes are appended to this format string.')
def cli(device, input, batch_input, raise_on_error, format_type, verbose, suffix, pdf_format):
    """ YALTAi is built as a group of command but only takes one command at the time: segment """
    ctx = click.get_current_context()
    if device != 'cpu':
        import torch
        try:
            torch.ones(1, device=device)
        except AssertionError as e:
            if raise_on_error:
                raise
            logger.error(f'Device {device} not available: {e.args[0]}.')
            ctx.exit(1)
    ctx.meta['device'] = device
    ctx.meta['input_format_type'] = format_type if format_type != 'pdf' else 'image'
    ctx.meta['raise_failed'] = raise_on_error
    ctx.meta['output_mode'] = "alto"  # Unlike Kraken, forces ALTO
    ctx.meta['verbose'] = verbose
    ctx.meta['steps'] = []
    log.set_logger(logger, level=30 - min(10 * verbose, 20))


@cli.result_callback()
def process_pipeline(subcommands, input, batch_input, suffix, verbose, format_type, pdf_format, **args):
    """
    Helper function calling the partials returned by each subcommand and
    placing their respective outputs in temporary files.
    """
    import glob
    import uuid
    import tempfile

    ctx = click.get_current_context()

    input = list(input)
    # expand batch inputs
    if batch_input and suffix:
        for batch_expr in batch_input:
            for in_file in glob.glob(batch_expr, recursive=True):
                input.append((in_file, '{}{}'.format(os.path.splitext(in_file)[0], suffix)))

    # parse pdfs
    if format_type == 'pdf':
        import pyvips

        if not batch_input:
            logger.warning('PDF inputs not added with batch option. Manual output filename will be ignored and `-o` utilized.')
        new_input = []
        num_pages = 0
        for (fpath, _) in input:
            doc = pyvips.Image.new_from_file(fpath, dpi=300, n=-1, access="sequential")
            if 'n-pages' in doc.get_fields():
                num_pages += doc.get('n-pages')

        with KrakenProgressBar() as progress:
            pdf_parse_task = progress.add_task('Extracting PDF pages', total=num_pages, visible=True if not ctx.meta['verbose'] else False)
            for (fpath, _) in input:
                try:
                    doc = pyvips.Image.new_from_file(fpath, dpi=300, n=-1, access="sequential")
                    if 'n-pages' not in doc.get_fields():
                        logger.warning('{fpath} does not contain pages. Skipping.')
                        continue
                    n_pages = doc.get('n-pages')

                    dest_dict = {'idx': -1, 'src': fpath, 'uuid': None}
                    for i in range(0, n_pages):
                        dest_dict['idx'] += 1
                        dest_dict['uuid'] = str(uuid.uuid4())
                        fd, filename = tempfile.mkstemp(suffix='.png')
                        os.close(fd)
                        doc = pyvips.Image.new_from_file(fpath, dpi=300, page=i, access="sequential")
                        logger.info(f'Saving temporary image {fpath}:{dest_dict["idx"]} to {filename}')
                        doc.write_to_file(filename)
                        new_input.append((filename, pdf_format.format(**dest_dict) + suffix))
                        progress.update(pdf_parse_task, advance=1)
                except pyvips.error.Error:
                    num_pages -= n_pages
                    progress.update(pdf_parse_task, total=num_pages)
                    logger.warning(f'{fpath} is not a PDF file. Skipping.')
        input = new_input
        ctx.meta['steps'].insert(0, {'category': 'preprocessing', 'description': 'PDF image extraction', 'settings': {}})

    for io_pair in input:
        ctx.meta['first_process'] = True
        ctx.meta['last_process'] = False
        ctx.meta['orig_file'] = io_pair[0]
        if 'base_image' in ctx.meta:
            del ctx.meta['base_image']
        try:
            tmps = [tempfile.mkstemp() for _ in subcommands[1:]]
            for tmp in tmps:
                os.close(tmp[0])
            fc = [io_pair[0]] + [tmp[1] for tmp in tmps] + [io_pair[1]]
            for idx, (task, input, output) in enumerate(zip(subcommands, fc, fc[1:])):
                if len(fc) - 2 == idx:
                    ctx.meta['last_process'] = True
                task(input=input, output=output)
        except Exception as e:
            logger.error(f'Failed processing {io_pair[0]}: {str(e)}')
            if ctx.meta['raise_failed']:
                raise
        finally:
            for f in fc[1:-1]:
                os.unlink(f)
            # clean up temporary PDF image files
            if format_type == 'pdf':
                logger.debug(f'unlinking {fc[0]}')
                os.unlink(fc[0])


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
def segment(ctx, model, text_direction, mask, yolo, ignore_lines, deskew, max_angle):
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

    return partial(segmenter, model, text_direction, mask, ctx.meta['device'], yolo, ignore_lines, deskew, max_angle)


if __name__ == "__main__":
    cli()
