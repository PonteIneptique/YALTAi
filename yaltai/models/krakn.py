from typing import Optional, Callable, Union, List, Dict, Any, Literal

import PIL
import logging
import uuid
import numpy as np
import shapely.geometry as geom
from kraken.blla import compute_segmentation_map, vec_lines

from kraken.containers import BaselineLine, Region, Segmentation
from kraken.lib.segmentation import (polygonal_reading_order, scale_regions, neural_reading_order, is_in_region)
from kraken.lib import vgsl
from kraken.lib.exceptions import KrakenInvalidModelException
from kraken.lib.util import get_im_str

logger = logging.getLogger(__name__)


def region_to_objects(regions: Dict[str, List[List[int]]]) -> Dict[str, List[Region]]:
    new_regions = {}
    for region_type, list_of_regions in regions.items():
        new_regions[region_type] = [
            Region(id=str(uuid.uuid4()), boundary=x, tags={'type': region_type})
            for x in list_of_regions
        ]
    return new_regions


def segment(im: PIL.Image.Image,
            text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr',
            mask: Optional[np.ndarray] = None,
            reading_order_fn: Callable = polygonal_reading_order,
            model: Union[List[vgsl.TorchVGSLModel], vgsl.TorchVGSLModel] = None,
            device: str = 'cpu',
            raise_on_error: bool = False,
            autocast: bool = False,
            regions: Optional[Dict[str, List[List[int]]]] = None,
            ignore_lignes: bool = False) -> Segmentation:
    r"""
    Segments a page into text lines using the baseline segmenter.

    Segments a page into text lines and returns the polyline formed by each
    baseline and their estimated environment.

    Args:
        im: Input image. The mode can generally be anything but it is possible
            to supply a binarized-input-only model which requires accordingly
            treated images.
        text_direction: Passed-through value for serialization.serialize.
        mask: A bi-level mask image of the same size as `im` where 0-valued
              regions are ignored for segmentation purposes. Disables column
              detection.
        reading_order_fn: Function to determine the reading order.  Has to
                          accept a list of tuples (baselines, polygon) and a
                          text direction (`lr` or `rl`).
        model: One or more TorchVGSLModel containing a segmentation model. If
               none is given a default model will be loaded.
        device: The target device to run the neural network on.
        raise_on_error: Raises error instead of logging them when they are
                        not-blocking
        autocast: Runs the model with automatic mixed precision

    Returns:
        A :class:`kraken.containers.Segmentation` class containing reading
        order sorted baselines (polylines) and their respective polygonal
        boundaries as :class:`kraken.containers.BaselineLine` records. The
        last and first point of each boundary polygon are connected.

    Raises:
        KrakenInvalidModelException: if the given model is not a valid
                                     segmentation model.
        KrakenInputException: if the mask is not bitonal or does not match the
                              image size.

    Notes:
        Multi-model operation is most useful for combining one or more region
        detection models and one text line model. Detected lines from all
        models are simply combined without any merging or duplicate detection
        so the chance of the same line appearing multiple times in the output
        are high. In addition, neural reading order determination is disabled
        when more than one model outputs lines.
    """
    # Unlike Kraken base implementation, we only accept Model and List of Models
    if isinstance(model, vgsl.TorchVGSLModel):
        model = [model]

    for nn in model:
        if nn.model_type != 'segmentation':
            raise KrakenInvalidModelException(f'Invalid model type {nn.model_type} for {nn}')
        if 'class_mapping' not in nn.user_metadata:
            raise KrakenInvalidModelException(f'Segmentation model {nn} does not contain valid class mapping')

    if ignore_lignes:
        return {'text_direction': text_direction,
                'type': 'baselines',
                'lines': [],
                'regions': regions,
                'script_detection': False}

    im_str = get_im_str(im)
    logger.info(f'Segmenting {im_str}')

    lines = []
    order = None
    regions = region_to_objects(regions)
    multi_lines = False
    # flag to indicate that multiple models produced line output -> disable
    # neural reading order
    for net in model:
        if 'topline' in net.user_metadata:
            loc = {None: 'center',
                   True: 'top',
                   False: 'bottom'}[net.user_metadata['topline']]
            logger.debug(f'Baseline location: {loc}')

        rets = compute_segmentation_map(im, mask, net, device, autocast=autocast)

        # We can't clear the heatmap of regions because it would mess up
        # print(rets)
        if "regions" in rets:
            del rets["regions"]

        # flatten regions for line ordering/fetch bounding regions
        line_regs = []
        suppl_obj = []
        for cls, regs in regions.items():
            line_regs.extend(regs)
            if rets['bounding_regions'] is not None and cls in rets['bounding_regions']:
                suppl_obj.extend(regs)

        # convert back to net scale
        suppl_obj = scale_regions([x.boundary for x in suppl_obj], 1/rets['scale'])
        line_regs = scale_regions([x.boundary for x in line_regs], 1/rets['scale'])

        _lines = vec_lines(**rets,
                           regions=line_regs,
                           text_direction=text_direction,
                           suppl_obj=suppl_obj,
                           topline=net.user_metadata['topline'] if 'topline' in net.user_metadata else False,
                           raise_on_error=raise_on_error)

        if 'ro_model' in net.aux_layers:
            logger.info(f'Using reading order model found in segmentation model {net}.')
            _order = neural_reading_order(lines=_lines,
                                          regions=regions,
                                          text_direction=text_direction[-2:],
                                          model=net.aux_layers['ro_model'],
                                          im_size=im.size,
                                          class_mapping=net.user_metadata['ro_class_mapping'])
        else:
            _order = None

        if _lines and lines or multi_lines:
            multi_lines = True
            order = None
            logger.warning('Multiple models produced line output. This is '
                           'likely unintended. Suppressing neural reading '
                           'order.')
        else:
            order = _order

        lines.extend(_lines)

    if len(rets['cls_map']['baselines']) > 1:
        script_detection = True
    else:
        script_detection = False

    # create objects and assign IDs
    blls = []
    _shp_regs = {}
    for reg_type, rgs in regions.items():
        for reg in rgs:
            _shp_regs[reg.id] = geom.Polygon(reg.boundary)

    # reorder lines
    logger.debug(f'Reordering baselines with main RO function {reading_order_fn}.')
    basic_lo = reading_order_fn(lines=lines, regions=_shp_regs.values(), text_direction=text_direction[-2:])
    lines = [lines[idx] for idx in basic_lo]

    for line in lines:
        line_regs = []
        for reg_id, reg in _shp_regs.items():
            line_ls = geom.LineString(line['baseline'])
            if is_in_region(line_ls, reg):
                line_regs.append(reg_id)
        blls.append(BaselineLine(id=str(uuid.uuid4()), baseline=line['baseline'], boundary=line['boundary'], tags=line['tags'], regions=line_regs))

    return Segmentation(text_direction=text_direction,
                        imagename=getattr(im, 'filename', None),
                        type='baselines',
                        lines=blls,
                        regions=regions,
                        script_detection=script_detection,
                        line_orders=[order] if order else [])



