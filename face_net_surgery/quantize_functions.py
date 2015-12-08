import math
import random

def round_number_hard(num, fixedPointList):
    '''
    Rounds num to closest number in fixedPointList
    :param num:
    :param fixedPointList:
    :return: quantized number
    '''
    low = 0
    high = len(fixedPointList)
    result = 0

    while low < high:
        midIdx = (low + high) / 2
        result = fixedPointList[midIdx]
        minDistance = abs(num - result)
        leftIdx = midIdx - 1
        rightIdx = midIdx + 1
        if leftIdx > -1 and abs(num - fixedPointList[leftIdx]) < minDistance:
            high = midIdx
        elif rightIdx < len(fixedPointList) and abs(num - fixedPointList[rightIdx]) < minDistance:
            low = midIdx + 1
        else:
            break

    return result
def round_number_stochastic(num, fixedPointList):
    '''
    Rounds num to closest number stochastically in fixedPointList
    :param num:
    :param fixedPointList:
    :return: quantized number
    '''
    low = 0
    high = len(fixedPointList)
    result = 0

    if num < fixedPointList[0]:
        return fixedPointList[0]
    if num > fixedPointList[-1]:
        return fixedPointList[-1]

    while low < high:
        midIdx = (low + high) / 2
        result = fixedPointList[midIdx]
        minDistance = abs(num - result)
        leftIdx = midIdx - 1
        rightIdx = midIdx + 1
        if leftIdx > -1 and abs(num - fixedPointList[leftIdx]) < minDistance:
            high = midIdx
        elif rightIdx < len(fixedPointList) and abs(num - fixedPointList[rightIdx]) < minDistance:
            low = midIdx + 1
        else:
            break

    # Since passed first two tasks, result must be between lowest and highest num
    if midIdx == len(fixedPointList) - 1:
        compareIdx = midIdx - 1
    elif midIdx == 0:
        compareIdx = midIdx + 1
    else:
        leftIdx = midIdx - 1
        rightIdx = midIdx + 1
        if abs(fixedPointList[leftIdx] - num) < abs(fixedPointList[rightIdx] - num):
            compareIdx = leftIdx
        else:
            compareIdx = rightIdx

    # decide result to be closest bin or second closest bin
    compareVal = fixedPointList[compareIdx]
    probClosest = abs(compareVal - num) / abs(result - compareVal)
    if random.random() < probClosest:
        return result
    else:
        return compareVal
def round_number(num, fixedPointList, stochastic = False):
    '''
    Rounds num to closest number in fixedPointList
    :param num:
    :param fixedPointList:
    :param stochastic: decides whether to apply stochastic rounding
    :return: quantized number
    '''
    if stochastic == True:
        return round_number_stochastic(num, fixedPointList)
    else:
        return round_number_hard(num, fixedPointList)
def fixed_point_list(a, b):
    '''
    A(a, b) : range = -2^a ~ 2^a - 2^-b
    :param a:
    :param b:
    :return:  list of all numbers possible in A(a, b), from smallest to largest
    '''
    fixedPointList = []
    numOfElements = 2**(a + b + 1)

    for i in range(numOfElements):
        fixedPointList.append( -(2**a) + (2**(-b))*i )

    return fixedPointList

def tri_section_points(a):
    '''
    Returns three points (-2^a / 2, 0, 2^a / 2)
    :param nIn: number of neurons in input layer (filter input neurons for conv layers)
    :param nOut: number of neurons in output layer (1 for conv layers)
    :return: list with 3 numbers
    '''
    triSectionPoints = [-(2**a) / 2, 0, (2**a) / 2]

    return triSectionPoints

def soft_quantize_rounding(num, a, stochastic = False):
    '''
    Does hard quantizing after soft quantizing is done (rounds number to -1 0 1)
    :param num: number to quantize
    :param a: During soft quantizing, params are rounded to (-2^a / 2, 0, 2^a / 2)
    :param stochastic: decides whether to apply stochastic rounding
    :return: quantized number
    '''
    if stochastic == False:
        if num < -(2**a) / 4:
            return -1
        elif num > (2**a) / 4:
            return 1
        else:
            return 0

# def tri_section_points(nIn, nOut):
#     '''
#     Tri section point initialization method in peihen's oral_rehearsal P.21
#     :param nIn: number of neurons in input layer (filter input neurons for conv layers)
#     :param nOut: number of neurons in output layer (1 for conv layers)
#     :return: list with 3 numbers
#     '''
#     triSectionPoints = [-2 * math.sqrt(6) / (3 * math.sqrt(nIn + nOut)), 0,
#                         2 * math.sqrt(6) / (3 * math.sqrt(nIn + nOut))]
#
#     return triSectionPoints
