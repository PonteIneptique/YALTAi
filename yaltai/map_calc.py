"""
Adapted from https://github.com/Cartucho/mAP and from pypi/mapcalc to retrieve precision and recall
"""

import numpy as np


class _ImageDetection:
    def __init__(self, score, label, boxes, used=False):
        self.boxes = boxes
        self.label = label
        self.score = score
        self.used = used


def _voc_ap(rec, prec):
    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.

    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """

    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """

    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def _check_dicts_for_content_and_size(ground_truth_dict: dict, result_dict: dict):
    """

    Checks if the content and the size of the arrays adds up.
    Raises and exception if not, does nothing if everything is ok.

    :param ground_truth_dict: dict with {boxes:, labels:}
    e.g.
    {
    'boxes':
        [[60., 80., 66., 92.],
         [59., 94., 68., 97.],
         [70., 87., 81., 94.],
         [8., 34., 10., 36.]],

    'labels':
        [2, 2, 3, 4]}
    :param result_dict: dict with {boxes:, labels:, scores:}
    e.g.
    {
    'boxes':
        [[57., 87., 66., 94.],
         [58., 94., 68., 95.],
         [70., 88., 81., 93.],
         [10., 37., 17., 40.]],

    'labels':
        [2, 3, 3, 4],

    'scores':
        [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
    :return:
    """
    if 'boxes' not in ground_truth_dict.keys():
        raise ValueError("ground_truth_dict expects the keys 'boxes' and 'labels'.")
    if 'labels' not in ground_truth_dict.keys():
        raise ValueError("ground_truth_dict expects the keys 'boxes' and 'labels'.")
    if 'boxes' not in result_dict.keys():
        raise ValueError("result_dict expects the keys 'boxes' and 'labels' and optionally 'scores'.")
    if 'labels' not in result_dict.keys():
        raise ValueError("result_dict expects the keys 'boxes' and 'labels' and optionally 'scores'.")

    if 'scores' not in result_dict.keys():
        result_dict['scores'] = [1] * len(result_dict['boxes'])

    if len(ground_truth_dict['boxes']) != len(ground_truth_dict['labels']):
        raise ValueError("The number of boxes and labels differ in the ground_truth_dict.")

    if not len(result_dict['boxes']) == len(result_dict['labels']) == len(result_dict['scores']):
        raise ValueError("The number of boxes, labels and scores differ in the result_dict.")


def calculate_map(ground_truth_dict: dict, result_dict: dict, iou_threshold: float):
    """
    mAP@[iou_threshold]

    :param ground_truth_dict: dict with {boxes:, labels:}
    e.g.
    {
    'boxes':
        [[60., 80., 66., 92.],
         [59., 94., 68., 97.],
         [70., 87., 81., 94.],
         [8., 34., 10., 36.]],

    'labels':
        [2, 2, 3, 4]}
    :param result_dict: dict with {boxes:, labels:, scores:}
    e.g.
    {
    'boxes':
        [[57., 87., 66., 94.],
         [58., 94., 68., 95.],
         [70., 88., 81., 93.],
         [10., 37., 17., 40.]],

    'labels':
        [2, 3, 3, 4],

    'scores':
        [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
    :param iou_threshold: minimum iou for which the detection counts as successful
    :return: mean average precision (mAP)
    """

    # checking if the variables have the correct keys

    _check_dicts_for_content_and_size(ground_truth_dict, result_dict)

    occurring_gt_classes = set(ground_truth_dict['labels'])
    unique, counts = np.unique(ground_truth_dict['labels'], return_counts=True)
    ground_truth_counter_per_class = dict(zip(unique, counts))
    count_true_positives = {}
    sum_average_precision = 0

    classes_scores = {

    }

    for class_index, class_name in enumerate(occurring_gt_classes):

        detections_with_certain_class = list()
        for idx in range(len(result_dict['labels'])):
            if result_dict['labels'][idx] == class_name:
                detections_with_certain_class.append(_ImageDetection(score=result_dict['scores'][idx],
                                                                     label=result_dict['labels'][idx],
                                                                     boxes=result_dict['boxes'][idx]))
        ground_truth_list = list()
        for idx in range(len(ground_truth_dict['labels'])):
            ground_truth_list.append(_ImageDetection(score=1,
                                                     label=ground_truth_dict['labels'][idx],
                                                     boxes=ground_truth_dict['boxes'][idx]))

        count_true_positives[class_name] = 0

        tp = [0] * len(detections_with_certain_class)
        fp = [0] * len(detections_with_certain_class)

        for i, elem in enumerate(detections_with_certain_class):
            ovmax = -1
            gt_match = -1

            bb = elem.boxes
            for j, elem in enumerate(ground_truth_list):
                if ground_truth_list[j].label == class_name:
                    bbgt = ground_truth_list[j].boxes
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = elem

            if ovmax >= iou_threshold:
                if not gt_match.used:
                    # true positive
                    tp[i] = 1
                    gt_match.used = True
                    count_true_positives[class_name] += 1
                    # update the ".json" file
                else:
                    # false positive (multiple detection)
                    fp[i] = 1
            else:
                # false positive
                fp[i] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val

        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / ground_truth_counter_per_class[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        average_precision, mean_recall, mean_precision = _voc_ap(rec[:], prec[:])
        classes_scores[class_index] = (
            average_precision,
            sum(mean_precision) / len(mean_precision),
            sum(mean_recall) / len(mean_recall)
        )

        sum_average_precision += average_precision

    mean_average_precision = sum_average_precision / len(occurring_gt_classes)
    return mean_average_precision, classes_scores


def calculate_map_range(ground_truth_dict: dict, result_dict: dict, iou_begin: float, iou_end: float, iou_step: float):
    """
    Gives mAP@[iou_begin:iou_end:iou_step], including iou_begin and iou_end.

    :param ground_truth_dict: dict with {boxes:, labels:}
    e.g.
    {
    'boxes':
        [[60., 80., 66., 92.],
         [59., 94., 68., 97.],
         [70., 87., 81., 94.],
         [8., 34., 10., 36.]],

    'labels':
        [2, 2, 3, 4]}
    :param result_dict: dict with {boxes:, labels:, scores:}
    e.g.
    {
    'boxes':
        [[57., 87., 66., 94.],
         [58., 94., 68., 95.],
         [70., 88., 81., 93.],
         [10., 37., 17., 40.]],

    'labels':
        [2, 3, 3, 4],

    'scores':
        [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
    :param iou_begin: first iou to evaluate
    :param iou_end: last iou to evaluate (included!)
    :param iou_step: step size
    :param allow_cut_off: If true, there will be no exception if the number of predictions is not the same than
    the number of ground truth values. Will cut off predictions with the least scores.
    :return: mean average precision
    """

    _check_dicts_for_content_and_size(ground_truth_dict, result_dict)

    iou_list = np.arange(iou_begin, iou_end + iou_step, iou_step)

    mean_average_precision_sum = 0.
    for iou in iou_list:
        mean_average_precision_sum += calculate_map(ground_truth_dict, result_dict, iou)

    return mean_average_precision_sum / len(iou_list)
