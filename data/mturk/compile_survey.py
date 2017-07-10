#!/bin/env python

import csv
import json
import random
import re

import math
import numpy
import operator

rgbstr_regex = re.compile("^\s*?rgb\(\s*?(\d+?)\s*?,\s*?(\d+?)\s*?,\s*?(\d+?)\)\s*?$")
hexstr_regex = re.compile("^#[0-9A-F]{6}$")


def hex_to_rgb(hex):
    if hex.startswith("#"):
        hex = hex[1:]

    i = int(hex, 16)
    r = (i >> 16) & 0xFF
    g = (i >> 8) & 0xFF
    b = i & 0xFF
    return (r/255.0, g/255.0, b/255.0)


def color_str_to_rgb(str):
    rgbstr_match = rgbstr_regex.match(str)
    hexstr_match = hexstr_regex.match(str)
    if hexstr_match is not None:
        return hex_to_rgb(str)
    elif rgbstr_match is not None:
        return (int(rgbstr_match.group(1))/255.0, int(rgbstr_match.group(2))/255.0, int(rgbstr_match.group(3))/255.0)
    else:
        return None

def rgb_to_color_str(rgb):
    r = str(int(255*rgb[0]))
    g = str(int(255*rgb[1]))
    b = str(int(255*rgb[2]))
    return "rgba("+r+","+g+","+b+",1.0)"

def distance(col1, col2):
    return math.sqrt(
        (
            (col1[0] - col2[0]) ** 2 +
            (col1[1] - col2[1]) ** 2 +
            (col1[2] - col2[2]) ** 2
        )
    )
worker_index = {}
color_index = {}
hit_index = {}

# read input csv
fp = open("mturk_results_raw.csv", "rb")
reader = csv.DictReader(fp)
next(reader, None)  # skip header
for line in reader:
    hit = line["HITId"]
    worker = line["WorkerId"]

    duration = line["AssignmentDurationInSeconds"]
    colors = json.loads(line["Input.colors"])

    # collect answers
    BATCH_SIZE = 20
    outputs = {}
    for i in range(0, len(colors)):
        converted = color_str_to_rgb(line["Answer." + str(i)])
        if converted is not None:
            outputs[colors[i]] = converted

    if worker not in worker_index:
        worker_index[worker] = {}

    for k in outputs:
        worker_index[worker][k] = outputs[k]

        if k not in color_index:
            color_index[k] = {}

        color_index[k][worker] = outputs[k]

# calculate each individual worker distance
worker_weights = {}
for worker in worker_index:
    distances = []
    for color in worker_index[worker]:
        worker_value = worker_index[worker][color]
        all_work = color_index[color]
        cur_distance = 0
        for w in all_work:
            work = all_work[w]
            cur_distance += distance(work, worker_value)
        if len(all_work) - 1 == 0:
            cur_distance = 0.0
        else:
            # the distance of the current work item is 0, so -1
            cur_distance = (cur_distance / (len(all_work) - 1.0))
        distances.append(cur_distance)
    avg_distance = math.fsum(distances)/len(distances)
    worker_weights[worker] = 1-(avg_distance/math.sqrt(3))**2

bad_percentile = 70
bad_distance = numpy.percentile(worker_weights.values(), bad_percentile)
print bad_distance

centers = {}
avg_distance_from_center = {}

for color in color_index:
    weighted_sum = [0, 0, 0]
    sum = [0,0,0]
    sum_of_weights = 0.0
    count = 0

    for worker in color_index[color]:
        work = color_index[color][worker]
        weight = worker_weights[worker]
        for i in range(0,3):
            weighted_sum[i] += work[i] * weight
            sum[i] += work[i]
        count += 1
        sum_of_weights += weight

    weighted_avg = [0,0,0]
    avg = [0,0,0]
    for i in range(0,3):
        weighted_avg[i] = weighted_sum[i] / sum_of_weights
        avg[i] = sum[i] / count

    centers[color] = weighted_avg

    weighted_distance = 0.0
    sum_of_weights = 0.0
    for worker in color_index[color]:
        work = color_index[color][worker]
        partial = distance(work, centers[color])

        sum_of_weights += worker_weights[worker]
        weighted_distance += partial * worker_weights[worker]

    weighted_distance /= sum_of_weights
    avg_distance_from_center[color] = weighted_distance

# order by avg_distance_from_center
sorted_avg_distance = sorted(avg_distance_from_center.items(), key=operator.itemgetter(1))

# print '''
# <html>
# <body>
# <table border=0>
# <tr><td>out</td><td>name</td><td>corr</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr>
# '''
# for (color, avg_distance) in sorted_avg_distance:
#     center = centers[color]
#
#     print "<tr>"
#     print "<td width='50px' style='background-color: "+rgb_to_color_str(center)+"'></td>"
#     print "<td>"+color+"</td>"
#     print "<td>"+str(avg_distance)+"</td></td>"
#     for worker in color_index[color]:
#         col = color_index[color][worker]
#         print "<td width='50px' style='background-color: " + rgb_to_color_str(col) + "'></td>"
#     print "</tr>"
# print '''
# </table>
# </body>
# </html>
# '''
#
PERCENTILE = 75
p = numpy.percentile(avg_distance_from_center.values(), PERCENTILE)
print "color,red,green,blue"
filtered = [color for color in avg_distance_from_center if avg_distance_from_center[color] < p]
random.shuffle(filtered)

for color in filtered:
    center = centers[color]
    print "\""+color + "\"," + str(int(255*center[0])) + "," + str(int(255*center[1])) + "," + str(int(255*center[2]))
