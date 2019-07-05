import os
import path
import glob
from PIL import Image, ImageDraw

# ground truth directory
gt_text_dir = "./ICDAR_15/textLocalization/test/gt" #"./ICDAR_15/textLocalization/train/gt"

# original images directory
image_dir = "./ICDAR_15/textLocalization/test/*.jpg" #"./ICDAR_15/textLocalization/train/*.jpg"
imgDirs = []
imgLists = glob.glob(image_dir)

# where to save the images with ground truth boxes
imgs_save_dir = "./ICDAR_15/textLocalization/test/ICDAR_with_GT" #"./ICDAR_15/textLocalization/train/ICDAR_with_GT"
if not os.path.exists(imgs_save_dir):
	os.mkdir(imgs_save_dir)

for item in imgLists:
	imgDirs.append(item)

for img_dir in imgDirs:
	img = Image.open(img_dir)
	dr = ImageDraw.Draw(img)
	# extract the basename of the image address and split it from its suffix.
	img_basename = os.path.basename(img_dir)
	(img_name, suffix) = os.path.splitext(img_basename)
	# open the ground truth text file use the basename of the associated image
	img_gt_text_name = "gt_" + img_name + ".txt"
	print(img_gt_text_name)
	# read the gt txt, split as line to constitute a list.
	bf = open(os.path.join(gt_text_dir, img_gt_text_name), encoding='utf-8-sig').read().splitlines()

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

	# save the gt labeled image
	img.save(os.path.join(imgs_save_dir, img_basename))