import cv2
import numpy as np


class AutoCropper:
    """
    Autocroppy, designed to remove black borders from scanned documents - https://github.com/gerwin3/autocroppy
    """
    img = None

    max_border_size = 300  # maximum border to remove after initial crop (so not guaranteed!)
    safety_margin = 4  # extra border removed to make sure we have no black pixels
    tolerance = 4  # maximum gray-value to consider black

    def __init__(self, img):
        self.img = img
        print(f"in auto_cropper: {np.shape(img)}")

    def __borders_left_top(self, img):
        c = self.max_border_size
        print(f"img: {np.shape(img)}")
        print(f"{c=}")
        print(f"{self.tolerance=}")
        while c > 0:
            if img[c, c] < self.tolerance and img[c - 1, c - 1] < self.tolerance and img[c - 2, c - 2] < self.tolerance:
                return c, c
            c -= 1
        return 0, 0

    def __borders_left_bottom(self, img):
        c = self.max_border_size
        while c > 0:
            if img[-c, c] < self.tolerance and img[-(c - 1), c - 1] < self.tolerance \
                    and img[-(c - 2), c - 2] < self.tolerance:
                return c, img.shape[0] - c
            c -= 1
        return 0, img.shape[0]

    def __borders_right_top(self, img):
        c = self.max_border_size
        while c > 0:
            if img[c, -c] < self.tolerance and img[c - 1, -(c - 1)] < self.tolerance and img[
                        c - 2, -(c - 2)] < self.tolerance:
                return img.shape[1] - c, c
            c -= 1
        return img.shape[1], 0

    def __borders_right_bottom(self, img):
        c = self.max_border_size
        while c > 0:
            if img[-c, -c] < self.tolerance and img[-(c - 1), -(c - 1)] < self.tolerance \
                    and img[-(c - 2), -(c - 2)] < self.tolerance:
                return img.shape[1] - c, img.shape[0] - c
            c -= 1
        return img.shape[1], img.shape[0]

    # crops the image to remove black borders on the side, this specific
    # algorithm is build to handle rounded corners in, for example, scanned
    # slides or pages
    def autocrop(self):

        # apply a tolerance on a gray version of the image to
        # select the non-black pixels
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours_result = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[0]

        # find the contour with the highest area, that will be
        # a slightly too big crop of what we need
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt

        # crop it like this so we can perform additional operations
        # to further narrow down the crop
        x, y, w, h = cv2.boundingRect(best_cnt)
        gray_crop = gray[y:y + h, x:x + w]
        color_crop = self.img[y:y + h, x:x + w]

        # this more complicated cropping algorithm takes the corners
        # of the image and finds the smallest rectangle that doesn't
        # cover any black borders
        left = 0
        right = gray.shape[1]
        top = 0
        bottom = gray.shape[0]

        x, y = self.__borders_left_top(gray_crop)
        if x > left:
            left = x
        if y > top:
            top = x

        x, y = self.__borders_left_bottom(gray_crop)
        if x > left:
            left = x
        if y < bottom:
            bottom = y

        x, y = self.__borders_right_top(gray_crop)
        if x < right:
            right = x
        if y > top:
            top = y

        x, y = self.__borders_right_bottom(gray_crop)
        if x < right:
            right = x
        if y < bottom:
            bottom = y

        # these are safety value, by removing another two pixels
        # from the sides we make sure to not include any black
        # pixels that don't belong
        left = max(left + self.safety_margin, 0)
        right = max(right - self.safety_margin, 0)
        top = max(top + self.safety_margin, 0)
        bottom = max(bottom - self.safety_margin, 0)

        # apply the calculated value in the final crop
        # and return!
        return color_crop[top:bottom, left:right]
