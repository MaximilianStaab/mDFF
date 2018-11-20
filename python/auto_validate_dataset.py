#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import shutil
import numpy as np
import cv2
from datetime import datetime


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="source directory", default="")
parser.add_argument("-f", "--first", help="first stack to validate")
parser.add_argument("-l", "--last", help="last stack to validate")
parser.add_argument("-d", "--del_incomplete", help="auto-delete stacks with missing/broken files or dirs (missing refocus data excluded)", action="store_true")
parser.add_argument("-da", "--del_all", help="auto-delete stacks with at least one failing heuristic (refocus data included)", action="store_true")
parser.add_argument("-o", "--output", help="output file path", type=argparse.FileType('w', 1), default="suspicious_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".txt")
args = parser.parse_args()

FS_PATH = os.path.join(args.input, "FocalStacks")
PC_PATH = os.path.join(args.input, "PointClouds")
RF_PATH = os.path.join(args.input, "Refocus")
OFH = args.output

SCALED_IM_SIZE = 40
BROKEN_THRESH = 0.5
MOVEMENT_THRESH = 0.99
CLOUD_THRESH = 0.01

def rem(name, reason, delete=False):
	if delete:
		print("Deleting "+name+" because "+reason+".")
		OFH.write("Deleting "+name+" because "+reason+".\n")
		for p in [FS_PATH, PC_PATH, RF_PATH]: shutil.rmtree(os.path.join(p, name))
	else:
		print("Check "+name+" because "+reason+".")
		OFH.write("Check "+name+" because "+reason+".\n")
	return False

def has10lines(fname):
	count = 0
	with open(fname) as f:
		count = sum([1 for line in f])
	return count

def similarity(fnames, is_refocus):
	pics = [cv2.resize(cv2.imread(fname), (SCALED_IM_SIZE, SCALED_IM_SIZE), interpolation = cv2.INTER_AREA) for fname in fnames]
	if not is_refocus:
		similarities = [pics_similarity(pic1, pic2) for pic1, pic2 in zip(pics[:-1], pics[1:])]
		if not any(similarities): return 0
		elif 2 in similarities: return 2
		else: return 1
	else:
		similarities = [pics_similarity2(pic1, pic2) for pic1, pic2 in zip(pics[:-1], pics[1:])]
		if not any(similarities): return 0
		else: return 2

def pics_similarity(pic1, pic2):
	sim_map = cv2.matchTemplate(pic1, pic2, cv2.TM_CCOEFF_NORMED)
	_, maxVal, _, _ = cv2.minMaxLoc(sim_map)
	if maxVal < BROKEN_THRESH:
		print(maxVal)
		return 2
	elif maxVal < MOVEMENT_THRESH:
		print(maxVal)
		return 1
	else:
		return 0

def pics_similarity2(pic1, pic2):
	sim_map = cv2.matchTemplate(pic1, pic2, cv2.TM_CCOEFF_NORMED)
	_, maxVal, _, _ = cv2.minMaxLoc(sim_map)
	if maxVal < BROKEN_THRESH:
		print(maxVal)
		return 2
	else:
		return 0

def clouds_sim(fnames):
	means = []
	for fname in fnames:
		cloud = np.genfromtxt(fname, delimiter=',', usecols=(0,1,2))
		av = np.average(cloud,0)
		means.append(av)
	dev = np.std(means, axis=0)
	are_similar = max(dev) <= CLOUD_THRESH
	if not are_similar: print(dev)
	return are_similar

def validate(name):
	if not os.path.isdir(os.path.join(PC_PATH, name)): return rem(name, "no point cloud dir", args.del_incomplete or args.del_all)
	if not os.path.isdir(os.path.join(RF_PATH, name)): return rem(name, "no refocus dir", args.del_all)
	if not all([os.path.isfile(os.path.join(FS_PATH, name, str(i)+".jpg")) for i in range(10)]):
		return rem(name, "<10 stack pics", args.del_incomplete or args.del_all)
	if not all([os.path.isfile(os.path.join(PC_PATH, name, str(i)+".txt")) for i in range(10)]):
		return rem(name, "<10 point clouds", args.del_incomplete or args.del_all)
	if not all([os.path.isfile(os.path.join(RF_PATH, name, str(i)+".jpg")) for i in range(10)]):
		return rem(name, "<10 refocus pics", args.del_all)
	if not os.path.isfile(os.path.join(RF_PATH, name, "dists.txt")): return rem(name, "no dists.txt", args.del_all)
	if not all([os.path.getsize(os.path.join(FS_PATH, name, str(i)+".jpg"))>0 for i in range(10)]):
		return rem(name, "stack pic broken: 0 Bytes", args.del_incomplete or args.del_all)
	if not all([os.path.getsize(os.path.join(PC_PATH, name, str(i)+".txt"))>0 for i in range(10)]):
		return rem(name, "point cloud broken: 0 Bytes", args.del_incomplete or args.del_all)
	if not all([os.path.getsize(os.path.join(RF_PATH, name, str(i)+".jpg"))>0 for i in range(10)]):
		return rem(name, "refocus pic broken: 0 Bytes", args.del_all)
	if not has10lines(os.path.join(RF_PATH, name, "dists.txt")): return rem(name, "dists.txt not has 10 lines", args.del_all)
	sim = similarity([os.path.join(FS_PATH, name, str(i)+".jpg") for i in range(10)], False)
	if sim == 2: return rem(name, "stack pic: probably broken", args.del_all)
	if sim == 1: return rem(name, "stack pic: movement detected", args.del_all)
	sim = similarity([os.path.join(RF_PATH, name, str(i)+".jpg") for i in range(10)], True)
	if sim == 2: return rem(name, "refocus pic: probably broken", args.del_all)
	if not clouds_sim([os.path.join(PC_PATH, name, str(i)+".txt") for i in range(10)]):
		return rem(name, "point clouds probably inconsistent", args.del_all)
	print(name+" accepted.")
	return True


if __name__ == "__main__":
	if not os.path.isdir(FS_PATH):
		print("No directory '"+FS_PATH+"' detected. Exiting...")
		sys.exit()

	names = [name for name in os.listdir(FS_PATH) if os.path.isdir(os.path.join(FS_PATH, name))]
	start, end = None, None
	if args.first is not None: start = names.index(args.first)
	if args.last is not None: end = names.index(args.last)+1
	for name in names[start:end]: validate(name)