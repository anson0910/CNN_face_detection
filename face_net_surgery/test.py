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
def round_number(num, fixedPointList):
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


fixedPointList = fixed_point_list(2, 0)

print fixedPointList
