from __future__ import print_function
import os
import cv2
import time
import codecs
import argparse
import xml.dom.minidom


def process_convert(rootName, txtName, output_dir, GT_Type=False):
	"""

	:param rootName:
	:param txtName:
	:param output_dir:
	:param GT_Type:
	:return:
	"""
	# Read the txt annotation file.
	file_name = os.path.join(rootName, txtName)
	image = txtName[3:-4] + '.jpg'
	img_name  = os.path.join(rootName[:-3], image)

	with codecs.open(file_name, 'r', encoding='utf-8') as f:
		lines = f.readlines()

	annotation_xml = xml.dom.minidom.Document()
	#<annotataion> aka root
	root = annotation_xml.createElement('annotation')
	annotation_xml.appendChild(root)

	#<folder>
	nodeFolder = annotation_xml.createElement('folder')
	root.appendChild(nodeFolder)
	#</folder>
	#<filename>
	nodeFilename = annotation_xml.createElement('filename')
	nodeFilename.appendChild(annotation_xml.createTextNode(image))
	root.appendChild(nodeFilename)
	#</filename>

	image = cv2.imread(img_name)
	if image is not None:
		#print(image)
		h, w, c = image.shape
	else:
		raise KeyError('img_name error:', img_name)

	# <size>
	nodeSize = annotation_xml.createElement('size')
	nodeWidth = annotation_xml.createElement('width')
	nodeHeight = annotation_xml.createElement('height')
	nodeDepth = annotation_xml.createElement('depth')
	nodeWidth.appendChild(annotation_xml.createTextNode(str(w)))
	nodeHeight.appendChild(annotation_xml.createTextNode(str(h)))
	nodeDepth.appendChild(annotation_xml.createTextNode(str(c)))
	nodeSize.appendChild(nodeWidth)
	nodeSize.appendChild(nodeHeight)
	nodeSize.appendChild(nodeDepth)
	root.appendChild(nodeSize)
	# </size>
	# extract coordinates
	for l in lines:
		l = l.encode('utf-8').decode('utf-8-sig')
		l = l.strip().split(',')
		print(l)
		label_name = str(l[-1])
		if label_name == '###':
			difficult = 1
		else:
			difficult = 0

		if GT_Type is True:
			# r0 = (xr01; yr01; xr02; yr02; hr0), rotated rectangle - GT
			xleft_text = str(int(float(l[1])))
			ytop_text = str(int(float(l[2])))
			xright_text = str(int(float(l[3])))
			ybottom_text = str(int(float(l[4])))

			x1_text = xleft_text
			x2_text = xright_text
			x3_text = xright_text
			x4_text = xleft_text

			y1_text = ytop_text
			y2_text = ytop_text
			y3_text = ybottom_text
			y4_text = ybottom_text

		else:
			# q0 = (xq01; yq01; xq02; yq02; xq03; yq03; xq04; yq04), quadrilateral - GT
			xs = [ int(l[i])  for i in (0, 2, 4, 6)]
			ys = [ int(l[i]) for i in (1, 3, 5, 7)]
			xmin_text = str(min(xs))
			ymin_text = str(min(ys))
			xmax_text = str(max(xs))
			ymax_text = str(max(ys))

			x1_text = str(xs[0])
			x2_text = str(xs[1])
			x3_text = str(xs[2])
			x4_text = str(xs[3])

			y1_text = str(ys[0])
			y2_text = str(ys[1])
			y3_text = str(ys[2])
			y4_text = str(ys[3])

		#<object>
		nodeObject = annotation_xml.createElement('object')
		#<difficult>
		nodeDifficult = annotation_xml.createElement('difficult')
		nodeDifficult.appendChild(annotation_xml.createTextNode(str(difficult)))
		nodeObject.appendChild(nodeDifficult)
		#</difficult>
		#<content>
		nodeContent = annotation_xml.createElement('content')
		nodeContent.appendChild(annotation_xml.createTextNode(label_name))
		nodeObject.appendChild(nodeContent)
		#</content>
		#<name>
		nodeName = annotation_xml.createElement('name')
		name = 'text' if difficult == 0 else 'none'
		nodeName.appendChild(annotation_xml.createTextNode(name))
		nodeObject.appendChild(nodeName)
		#</name>
		#<bndbox>
		nodeBndbox = annotation_xml.createElement('bndbox')
		#<coordinates x1, y1, x2, y2, x3, y3, x4, y4, xmin, ymin, xmax, ymax>
		nodexmin = annotation_xml.createElement('xmin')
		nodexmin.appendChild(annotation_xml.createTextNode(xmin_text))
		nodeymin = annotation_xml.createElement('ymin')
		nodeymin.appendChild(annotation_xml.createTextNode(ymin_text))
		nodexmax = annotation_xml.createElement('xmax')
		nodexmax.appendChild(annotation_xml.createTextNode(xmax_text))
		nodeymax = annotation_xml.createElement('ymax')
		nodeymax.appendChild(annotation_xml.createTextNode(ymax_text))

		nodex1 = annotation_xml.createElement('x1')
		nodex1.appendChild(annotation_xml.createTextNode(x1_text))
		nodex2 = annotation_xml.createElement('x2')
		nodex2.appendChild(annotation_xml.createTextNode(x2_text))
		nodex3 = annotation_xml.createElement('x3')
		nodex3.appendChild(annotation_xml.createTextNode(x3_text))
		nodex4 = annotation_xml.createElement('x4')
		nodex4.appendChild(annotation_xml.createTextNode(x4_text))

		nodey1 = annotation_xml.createElement('y1')
		nodey1.appendChild(annotation_xml.createTextNode(y1_text))
		nodey2 = annotation_xml.createElement('y2')
		nodey2.appendChild(annotation_xml.createTextNode(y2_text))
		nodey3 = annotation_xml.createElement('y3')
		nodey3.appendChild(annotation_xml.createTextNode(y3_text))
		nodey4 = annotation_xml.createElement('y4')
		nodey4.appendChild(annotation_xml.createTextNode(y4_text))

		nodeBndbox.appendChild(nodex1)
		nodeBndbox.appendChild(nodey1)
		nodeBndbox.appendChild(nodex2)
		nodeBndbox.appendChild(nodey2)
		nodeBndbox.appendChild(nodex3)
		nodeBndbox.appendChild(nodey3)
		nodeBndbox.appendChild(nodex4)
		nodeBndbox.appendChild(nodey4)
		nodeBndbox.appendChild(nodexmin)
		nodeBndbox.appendChild(nodeymin)
		nodeBndbox.appendChild(nodexmax)
		nodeBndbox.appendChild(nodeymax)

		#<coordinates x1, y1, x2, y2, x3, y3, x4, y4, xmin, ymin, xmax, ymax>
		nodeObject.appendChild(nodeBndbox)
		#</bndbox>
		root.appendChild(nodeObject)
		#</object>

	xml_path = os.path.join(output_dir, txtName[:-4] + '.xml')
	fp = open(xml_path, 'w')
	annotation_xml.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
	return True


def get_all_txt(directory, split_flag, logs_dir, output_dir, GT_Type=False):
	count = 0
	xml_path_list = []
	img_path_list = []
	if output_dir is not None and not os.path.exists(output_dir):
		os.makedirs(output_dir)

	start_time = time.time()
	for root,dirs,files in os.walk(directory):
		for file in files:
			if file.split('.')[-1] == 'txt':
				xml_path = os.path.join(root, file[:-4] + '.xml')
				img_path = os.path.join(root[:-3], file[3:-4] + '.jpg')

				if output_dir:
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					save_xml_path = os.path.join(output_dir, file[:-4] + '.xml')
				else:
					save_xml_path = xml_path

				if process_convert(root, file, output_dir, GT_Type):
					xml_path_list.append('{},{}\n'.format(img_path, save_xml_path))
					img_path_list.append('{}\n'.format(img_path))
				count += 1
				print(count, img_path)
				if count % 1000 == 0:
					print(count, time.time() - start_time)
	save_to_text(img_path_list, xml_path_list, count, split_flag, logs_dir)
	print('all over:', count)
	print('time:', time.time() - start_time, '\n')


def save_to_text(img_path_list, xml_path_list, count, split_flag, logs_dir):
	if split_flag == 'yes':
		train_num = int(count / 10. * 9.)
	else:
		train_num = count
	print('train img count {0}'.format(train_num))
	if not os.path.exists(logs_dir):
		os.makedirs(logs_dir)

	with codecs.open(
			os.path.join(logs_dir, 'train_xml.txt'), 'w', encoding='utf-8') as f_xml, codecs.open(
				os.path.join(logs_dir, 'train.txt'), 'w', encoding='utf-8') as f_txt:
					f_xml.writelines(xml_path_list[:train_num])
					f_txt.writelines(img_path_list[:train_num])

	if split_flag == 'yes':
		print('test img count {0}'.format(count - train_num))
		with codecs.open(
				os.path.join(logs_dir, 'test_xml.txt'), 'w', encoding='utf-8') as f_xml, codecs.open(
					os.path.join(logs_dir, 'test.txt'), 'w', encoding='utf-8') as f_txt:
						f_xml.writelines(xml_path_list[train_num:])
						f_txt.writelines(img_path_list[train_num:])


if  __name__ == '__main__':
	parser = argparse.ArgumentParser(description='icdar15 generate xml tools')
	parser.add_argument('--in_dir', '-i', default='./datasets/ICDAR_15/textLocalization/train', type=str,
	                    help='where to load training set and its gt txt files')
	parser.add_argument('--split_flag', '-s', default='no', type=str,
	                    help='whether or not to split the datasets')
	parser.add_argument('--save_logs', '-l', default='logs', type=str,
	                    help='whether to save train_xml.txt')
	parser.add_argument('--output_dir', '-o', default='./datasets/ICDAR_15/textLocalization/train/xml', type=str,
	                    help='where to save xmls')
	args = parser.parse_args()

	directory = args.in_dir
	split_flag = args.split_flag
	logs_dir = args.save_logs
	output_dir = args.output_dir

	get_all_txt(directory, split_flag, logs_dir, output_dir, False)
