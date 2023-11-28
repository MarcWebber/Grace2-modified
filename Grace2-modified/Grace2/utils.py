# 额外贪心算法获取排序
from __future__ import print_function
from collections import Counter
from itertools import chain


def get_apfd(test_sort, mutant_matrix):
    """
    根据变异矩阵获取某测试排序的APDF值
    :param test_sort: 测试用例排序
    :param mutant_matrix: 变异矩阵
    :return: apfds：每组apfd值集合, avg_apfd：apfds集合的平均值
    """
#    mutant_matrix = [mutant_matrix]
    b = []
    apfds = []
    for i in range(0, len(mutant_matrix), 5):
        b.append(mutant_matrix[i:i + 5])
    # 保留100组
    if len(b) > 100:
        del b[100:len(b)]
    for temp_matrix in b:
        temp_apfd = gen_apfd_value(temp_matrix, test_sort, "")
        apfds.append(temp_apfd)
    sum_apfd = 0
    for apfd in apfds:
        sum_apfd += apfd
    if len(apfds) == 0:
        avg_apfd = 0
    else:
        avg_apfd = sum_apfd / len(apfds)
    return apfds, avg_apfd

def calc_apfd(values, m, n):
    return 1-(sum(values)/(m*n*1.0))+(1/(2.0*n))

def gen_apfd_value(fault_matrix, testsPrio, programeName):
    """
    根据错误矩阵获取某测试用例排序的APFD
    :param fault_matrix:
    :param testsPrio:
    :param programeName:
    :return:
    """
    mutLived = 0
    apfd_calc = []

    for single_fault in fault_matrix:
        # 错误矩阵中为1的下标
        indexs = [index for index, value in enumerate(single_fault) if value == 1]
        # 排除没有用例或者20%以上用例可以检测的错误版本
        if len(indexs) == 0:
            mutLived += 1
            continue
        for idx, i in enumerate(testsPrio):
            if i in indexs: # 找到最先发现该错误的用例下标，记录并进入下一个错误版本检索
                apfd_calc.append(idx+1)
                break
    if len(fault_matrix)-mutLived <= 0 or len(testsPrio) <= 0:  # 如果没有错误版本可以满足或者测试用例数量为0
        print(programeName, "：没有错误满足", len(testsPrio), len(fault_matrix)-mutLived)
        return 0.0
    else:
        return calc_apfd(apfd_calc, len(fault_matrix)-mutLived, len(testsPrio)) #,apfd_calc


def get_apfdc(test_sort, mutant_matrix, time):
    """
    根据变异矩阵获取某测试排序的APDFC值
    :param test_sort: 测试用例排序
    :param mutant_matrix: 变异矩阵
    :para time: 测试用例的时间开销
    :return: apfdcs：每组apfdc值集合, avg_apfdc：apfdcs集合的平均值
    """
#    mutant_matrix = [mutant_matrix]
    b = []
    apfdcs = []
    for i in range(0, len(mutant_matrix), 5):
        b.append(mutant_matrix[i:i + 5])
    # 保留100组
    if len(b) > 100:
        del b[100:len(b)]
    for temp_matrix in b:
        temp_apfdc = gen_apfdc_value(temp_matrix, test_sort, time)
        apfdcs.append(temp_apfdc)
    sum_apfdc = 0
    for apfdc in apfdcs:
        sum_apfdc += apfdc
    if len(apfdcs) == 0:
        avg_apfdc = 0
    else:
        avg_apfdc = sum_apfdc / len(apfdcs)
    return apfdcs, avg_apfdc

def gen_apfdc_value(fault_matrix, testsPrio, time):
    """
    根据错误矩阵获取某测试用例排序的APFDC
    :param fault_matrix:
    :param testsPrio:
    :param time:
    :return:
    """
    mutLived = 0
    apfdc_calc = []

    for single_fault in fault_matrix:
        # 错误矩阵中为1的下标
        indexs = [index for index, value in enumerate(single_fault) if value == 1]
        # 排除没有用例或者20%以上用例可以检测的错误版本
        if len(indexs) == 0:
            mutLived += 1
            continue
        for idx, i in enumerate(testsPrio):
            if i in indexs: # 找到最先发现该错误的用例下标，记录并进入下一个错误版本检索
                apfdc_calc.append(idx+1)
                break
    if len(fault_matrix)-mutLived <= 0 or len(testsPrio) <= 0:  # 如果没有错误版本可以满足或者测试用例数量为0
        print("：没有错误满足", len(testsPrio), len(fault_matrix)-mutLived)
        return 0.0
    else:
        t1 = 0
        for i in range(len(fault_matrix)):
            t2 = 0
            for j in range(int(apfdc_calc[i])-1, len(fault_matrix[0])):
                t2 += time[testsPrio[j]]
            t1 += t2 - 0.5 * time[int(apfdc_calc[i])-1]
        apfdc = t1 / (sum(time) * len(fault_matrix))
        return apfdc
