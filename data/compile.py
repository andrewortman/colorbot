#!/usr/bin/env python

TEST_SIZE=500

import glob
import random

allcolors = []
for file in glob.glob("./*/db.csv"):
	fp = open(file, "r")
	lines = [line.strip() for line in fp.readlines()]
	allcolors += lines

random.shuffle(allcolors)

test_file = open("test.csv", "w")
train_file = open("train.csv", "w")

test_file.writelines("\n".join(allcolors[0:TEST_SIZE+1]))
train_file.writelines("\n".join(allcolors[TEST_SIZE+1:]))

test_file.close()
train_file.close()
