from typing import Tuple, List
from PIL import Image
import fast_deskew
import cv2
import numpy as np


def deskew(image: str) -> Tuple[Image.Image, float]:
    _, best_angle = fast_deskew.deskew_image(image, False)
    img = Image.open(image)
    return img.rotate(best_angle), best_angle


def rotatebox(bbox: List[List[int]], image: Image.Image, angle: float):
    # https://gist.githubusercontent.com/Joanne03/5941a9b4db4fa7c652a2d7f67b11a09b/raw/f99b2f18ad0fbe680ac631e531782b957f158def/rotate_bbox.py
    height, width = image.size
    image_center_x, image_center_y = width // 2, height // 2

    rotated_bbox = []

    for i, coord in enumerate(bbox):
      rot_matrix = cv2.getRotationMatrix2D((image_center_x, image_center_y), angle, 1.0)
      cosinus, sinus = abs(rot_matrix[0, 0]), abs(rot_matrix[0, 1])
      new_width = int((height * sinus) + (width * cosinus))
      new_height = int((height * cosinus) + (width * sinus))
      rot_matrix[0, 2] += (new_width / 2) - image_center_x
      rot_matrix[1, 2] += (new_height / 2) - image_center_y
      v = [coord[0], coord[1], 1]  # ?
      adjusted_coord = np.dot(rot_matrix, v)
      rotated_bbox.append((int(adjusted_coord[0]), int(adjusted_coord[1])))

    return rotated_bbox
