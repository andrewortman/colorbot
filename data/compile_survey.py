#!/usr/bin/env python
import glob
import random
import csv
import json

def write_csv(name, colors):
	fp = open(name, "w")
	writer = csv.writer(fp)
	writer.writerow(["colors"])
	for color in colors:
		writer.writerow(color)
	fp.close()

def write_vocab(name, vocab):
	keys = vocab.keys()
	keys.sort()
	fp = open(name, "wb")
	fp.write("\n".join(keys))
	fp.close()

allcolors = []
vocab = {}
color_names = {}

duplicates = 0
for file in glob.glob("../scraped/*/db.csv"):
	fp = open(file, "rb")
	reader = csv.reader(fp)
	for line in reader:
		color = line[0]

		red = line[1]
		green = line[2]
		blue = line[3]

		if color in color_names:
			color_names[color] += 1
			print "duplicate: " + color + ", " + str(color_names[color])
			continue
		else:
			color_names[color] = 1
		allcolors.append(color)
	fp.close()

random.shuffle(allcolors)

survey_colors = []
BATCH_SIZE=20
for i in range(0, len(allcolors), BATCH_SIZE):
    minidx = i
    maxidx = i+BATCH_SIZE
    if maxidx >= len(allcolors):
        maxidx = len(allcolors)-1

    slice = allcolors[minidx:maxidx]
    print slice
    survey_colors.append([json.dumps(slice)])

write_csv("mturk_input.csv", survey_colors)
