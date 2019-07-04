import os
import path
import glob
from PIL import Image, ImageDraw

# ground truth directory
gt_text_dir = "./ICDAR_15/textLocalization/train/gt"

# original images directory
image_dir = "./ICDAR_15/textLocalization/train/*.jpg"
imgDirs = []
imgLists = glob.glob(image_dir)

# where to save the images with ground truth boxes
imgs_save_dir = "/home/chenxp/Documents/Hitachi/ICDAR_2015/ICDAR_with_GT"

for item in imgLists:
    imgDirs.append(item)

for img_dir in imgDirs:
    img = Image.open(img_dir)
    dr = ImageDraw.Draw(img)

    img_basename = os.path.basename(img_dir)
    (img_name, temp2) = os.path.splitext(img_basename)
    # open the ground truth text file
    img_gt_text_name = "gt_" + img_name + ".txt"
    print(img_gt_text_name)

    bf = open(os.path.join(gt_text_dir, img_gt_text_name)).read().decode("utf-8-sig").encode("utf-8").splitlines()

    for idx in bf:
        rect = []
        spt = idx.split(',')
        rect.append(float(spt[0]))
        rect.append(float(spt[1]))
        rect.append(float(spt[2]))
        rect.append(float(spt[3]))
        rect.append(float(spt[4]))
        rect.append(float(spt[5]))
        rect.append(float(spt[6]))
        rect.append(float(spt[7]))

        # draw the polygon with (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4)
        dr.polygon((rect[0], rect[1], rect[2], rect[3], rect[4], rect[5], rect[6], rect[7]), outline="red")

    img.save(os.path.join(imgs_save_dir, img_basename))