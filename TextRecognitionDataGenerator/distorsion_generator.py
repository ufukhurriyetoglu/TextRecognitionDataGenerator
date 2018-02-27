import cv2
import math
import os
import random
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter

class DistorsionGenerator(object):
    @classmethod
    def apply_func_distorsion(cls, image, vertical, horizontal, max_offset, func):
        """
        """

        # Nothing to do!
        if not vertical and not horizontal:
            return image

        rgb_image = image.convert('RGB')
        
        img_arr = np.array(rgb_image)

        vertical_offsets = [func(i) for i in range(img_arr.shape[1])]
        horizontal_offsets = [func(i) for i in range(img_arr.shape[0] + (max(vertical_offsets) - min(min(vertical_offsets), 0)))]

        new_img_arr = np.zeros((
                          img_arr.shape[0] + (2 * max_offset if vertical else 0),
                          img_arr.shape[1] + (2 * max_offset if horizontal else 0),
                          3
                      ))

        if vertical:
            column_height = img_arr.shape[0]
            for i, o in enumerate(vertical_offsets):
                new_img_arr[max_offset+o:column_height+max_offset+o, i, :] = img_arr[:, i, :]

        if horizontal:
            row_width = img_arr.shape[1]
            for i, o in enumerate(horizontal_offsets):
                new_img_arr[i, max_offset+o:row_width+max_offset+o,:] = (new_img_arr[i, max_offset:row_width+max_offset, :] if vertical else img_arr[i, :, :])

        return Image.fromarray(np.uint8(new_img_arr))

    @classmethod
    def sin(cls, image, vertical=False, horizontal=False, max_offset=10):
        """
            Apply a sinus distorsion on one or both of the specified axis
        """

        return cls.apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: int(math.sin(math.radians(x)) * max_offset)))

    @classmethod
    def cos(cls, image, vertical=False, horizontal=False, max_offset=10):
        """
            Apply a cosine distorsion on one or both of the specified axis
        """

        return cls.apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: int(math.cos(math.radians(x)) * max_offset)))

    @classmethod
    def tan(cls, image, vertical=False, horizontal=False, max_offset=10):
        """
            Apply a tangent distorsion on one or both of the specified axis
        """

        return cls.apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: int(math.tan(math.radians(x)) * max_offset)))
    
    @classmethod
    def random(cls, image, vertical=False, horizontal=False, max_offset=5):
        """
            Apply a random distorsion on one or both of the specified axis
        """

        return cls.apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: random.randint(0, max_offset)))

