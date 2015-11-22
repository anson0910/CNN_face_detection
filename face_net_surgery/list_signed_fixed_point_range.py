'''
List ranges of signed fixed-point binary representations
'''
import numpy as np
import sys
from quantize_functions import *

file_write = open('signed_fixed_point_range.txt', 'w')
sys.stdout = file_write

for a in range(-10, 20):
    b = 2 - a
    fixedPointList = fixed_point_list(a, b)
    print "(" + str(a) + ", " + str(b) + ")  " \
            + str(fixedPointList[0]) + " ~ " + str(fixedPointList[-1])
