"""Evaluation metrics definitions."""


def intersection_over_union(boxA, boxB) -> float:
    """
    Computes the Intersection Over Union metric for 2 bounding boxes.
    :param boxA: tuple in a form (upper_left_x, upper_left_y, width, height)
    :param boxB: tuple in a form (upper_left_x, upper_left_y, width, height)
    :return: intersection over union
    """
    if boxA[3] == 0 or boxB[3] == 0:
        return None
    if boxA[0] + boxA[2] <= boxB[0] or boxB[0] + boxB[2] <= boxA[0]:
        return 0
    if boxA[1] + boxA[3] <= boxB[1] or boxB[1] + boxB[3] <= boxA[1]:
        return 0
    x_upper_left = max(boxA[0], boxB[0])
    y_upper_left = max(boxA[1], boxB[1])
    x_bottom_right = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    y_bottom_right = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    intersection_area = abs(max((x_bottom_right - x_upper_left) * (y_bottom_right - y_upper_left), 0))
    if intersection_area == 0:
        return 0
    boxA_area = abs(boxA[2] * boxA[3])
    boxB_area = abs(boxB[2] * boxB[3])
    iou = intersection_area / (boxA_area + boxB_area - intersection_area)
    return iou


def area_ratio(bbox_true, bbox_pred):
    """
    Computes the ratio of areas of 2 bboxes.
    :param bbox_true: tuple in a form (upper_left_x, upper_left_y, width, height)
    :param bbox_pred: tuple in a form (upper_left_x, upper_left_y, width, height)
    :return: area of bbox_pred / area of bbox_true
    """
    if bbox_pred[3] == 0:
        return None
    bbox_true_area = bbox_true[2] * bbox_true[3]
    bbox_pred_area = bbox_pred[2] * bbox_pred[3]
    return bbox_pred_area / bbox_true_area
