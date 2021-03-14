import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import argparse


attr_list = [
    "skin",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "eye_g",
    "l_ear",
    "r_ear",
    "ear_r",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "neck",
    "neck_l",
    "cloth",
    "hair",
    "hat",
]
attr_dict = dict([(i + 1, name) for i, name in enumerate(attr_list)])


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--img-path", default="imgs/116.jpg")
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def makeup(image, parsing, part_name="skin", color=[230, 50, 20]):
    if part_name not in attr_dict:
        return image
    part = attr_dict[part_name]
    b, g, r = color  # [10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed


if __name__ == "__main__":
    args = parse_args()

    image_path = args.img_path
    cp = "cp/79999_iter.pth"

    image = cv2.imread(image_path)
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    part_and_color = {
        # part_name: [B, G, R]
        "hair": [230, 50, 20],
        "skin": [130, 169, 240],
    }

    for part, color in part_and_color.items():
        image = makeup(image, parsing, part, color)

    cv2.imshow("image", cv2.resize(ori, (512, 512)))
    cv2.imshow("color", cv2.resize(image, (512, 512)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
