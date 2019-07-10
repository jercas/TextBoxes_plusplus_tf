# -*- coding: utf-8 -*-
import os
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Delete redundancy gt xml files in dataset directory.  ps: You may don't need the script!")
	parser.add_argument("--path", '-p', default='./datasets/ICDAR_15/textLocalization/train/gt', type=str,
	                    help='the path of the directory which redundancy files within you want to process')
	args = parser.parse_args()

	path = args.path
	if not os.path.exists(path):
		print("error path, can't find it!")
	else:
		for root, dirs, files in os.walk(path):
			for file in files:
				if file.split('.')[-1] == 'xml':
					absPath = os.path.join(root, file)
					print('Redundancy xml - {0} detected in {1}'.format(file, absPath))
					os.remove(absPath)