from typing import Optional, Callable, Union, List, Dict, Any

import PIL
import logging
import numpy as np
from kraken.blla import compute_segmentation_map, vec_regions, vec_lines

from kraken.lib.segmentation import (polygonal_reading_order, scale_regions)
from kraken.lib import vgsl
from kraken.lib.exceptions import KrakenInvalidModelException
from kraken.lib.util import get_im_str

logger = logging.getLogger(__name__)


def segment(im: PIL.Image.Image,
            text_direction: str = 'horizontal-lr',
            mask: Optional[np.ndarray] = None,
            reading_order_fn: Callable = polygonal_reading_order,
            model: Union[List[vgsl.TorchVGSLModel], vgsl.TorchVGSLModel] = None,
            device: str = 'cpu',
            regions: Optional[Dict[str, List[List[int]]]] = None,
            ignore_lignes: bool = False) -> Dict[str, Any]:
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
        regions: Regions computed using YOLOv5

    Returns:
        A dictionary containing the text direction and under the key 'lines' a
        list of reading order sorted baselines (polylines) and their respective
        polygonal boundaries. The last and first point of each boundary polygon
        are connected.

        .. code-block::
           :force:

            {'text_direction': '$dir',
             'type': 'baseline',
             'lines': [
                {'baseline': [[x0, y0], [x1, y1], ..., [x_n, y_n]], 'boundary': [[x0, y0, x1, y1], ... [x_m, y_m]]},
                {'baseline': [[x0, ...]], 'boundary': [[x0, ...]]}
              ]
              'regions': [
                {'region': [[x0, y0], [x1, y1], ..., [x_n, y_n]], 'type': 'image'},
                {'region': [[x0, ...]], 'type': 'text'}
              ]
            }

    Raises:
        KrakenInvalidModelException: if the given model is not a valid
                                     segmentation model.
        KrakenInputException: if the mask is not bitonal or does not match the
                              image size.
    """
    if isinstance(model, vgsl.TorchVGSLModel):
        model = [model]

    for nn in model:
        if nn.model_type != 'segmentation':
            raise KrakenInvalidModelException(f'Invalid model type {nn.model_type} for {nn}')
        if 'class_mapping' not in nn.user_metadata:
            raise KrakenInvalidModelException(f'Segmentation model {nn} does not contain valid class mapping')

    im_str = get_im_str(im)
    logger.info(f'Segmenting {im_str}')

    if ignore_lignes:
        return {'text_direction': text_direction,
                'type': 'baselines',
                'lines': [],
                'regions': regions,
                'script_detection': False}

    for net in model:
        if 'topline' in net.user_metadata:
            loc = {None: 'center',
                   True: 'top',
                   False: 'bottom'}[net.user_metadata['topline']]
            logger.debug(f'Baseline location: {loc}')

        rets = compute_segmentation_map(im, mask, net, device)

        # We can't clear the heatmap of regions because it would mess up

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
        suppl_obj = scale_regions(suppl_obj, 1/rets['scale'])
        line_regs = scale_regions(line_regs, 1/rets['scale'])
        lines = vec_lines(**rets,
                          regions=line_regs,
                          reading_order_fn=reading_order_fn,
                          text_direction=text_direction,
                          suppl_obj=suppl_obj,
                          topline=net.user_metadata['topline'] if 'topline' in net.user_metadata else False)

    if len(rets['cls_map']['baselines']) > 1:
        script_detection = True
    else:
        script_detection = False

    return {'text_direction': text_direction,
            'type': 'baselines',
            'lines': lines,
            'regions': regions,
            'script_detection': script_detection}
