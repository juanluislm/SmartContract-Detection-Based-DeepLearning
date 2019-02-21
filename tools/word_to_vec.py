# -*- coding:utf-8 -*-
import re
import numpy as np
from collections import Counter
from all_word_test import all_word_match


# 假设要读取文件名为aa，位于当前路径
fname = 'contract/contract10/9000.sol'


def word_to_vec(filename):
    with open(filename) as f:
        s = f.read()

    s1 = s.replace('\n', ' ').split(' ')
    s2 = " ".join(s1)
    result = re.findall('[a-zA-Z_]+', s2)
    counter = Counter(result)
    word_to_list = []
    for k, w in enumerate(counter.most_common()):
        word_to_list.append(w)

    all_word_results = all_word_match()

    # list -> array
    word_to_list_array = np.array(word_to_list)
    all_word_results_array = np.array(all_word_results)

    vec_result = ['0'] * 4500
    for i in range(4500):
        for j in range(len(word_to_list)):
            if word_to_list[j][0] == all_word_results[i]:
                vec_result[i] = str(word_to_list[j][1])
    vec_result_str = " ".join(vec_result)
    # print(vec_result_str)
    return vec_result_str


if __name__ == '__main__':
    word_to_vec(fname)
