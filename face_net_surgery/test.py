from quantize_functions import *

fixedPointList = fixed_point_list(2, 0)

print fixedPointList

closestTimes = 0
secondClosestTimes = 0

for curTry in range(10000):
    roundedNum = round_number(2.7, fixedPointList, True)
    if roundedNum == 3:
        closestTimes += 1
    else:
        secondClosestTimes += 1

print "Closest bin appeared " + str(closestTimes) + " times."
print "Second closest bin appeared " + str(secondClosestTimes) + " times."
