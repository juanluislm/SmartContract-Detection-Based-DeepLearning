"""
9000.sol 4 1 means contractname type frequency
"""
import os
from word_to_vec import word_to_vec

fileDirName = 'contract/contract10'


# 将合约漏洞类别转化为list
def read_vul_type():
    filepath = 'vulnerability/test.txt'
    f = open(filepath, 'r')
    result = []
    for line in f.readlines():
        line = line.strip()  # 去掉每行头尾空白
        if not len(line) or line.startswith('#'):
            continue
        line = line.split(' ')
        result.append(line)
    result.sort()
    print(result)
    return result


# 将合约漏洞类别向量化
def vul_type_to_vec():
    result = read_vul_type()
    word_to_vec = [0] * 14
    for i in range(len(result)):
        if result[i][0] != result[i - 1][0]:
            word_to_vec = [0] * 14
        if len(result[i]) > 1 and result[i][0] == result[i - 1][0]:
            word_to_vec[int(result[i][1]) - 1] = 1
            word_all_vec = str(word_to_vec)
            open('example_results/vec-result3.txt', 'a').write(result[i][0] + ' ' + '%s' % ''.join(word_all_vec) + '\n')
        if len(result[i]) == 1:
            word_to_vec[13] = 1
            word_all_vec = str(word_to_vec)
            open('example_results/vec-result3.txt', 'a').write(result[i][0] + ' ' + '%s' % ''.join(word_all_vec) + '\n')
            word_to_vec = [0] * 14


# 过滤没有用的向量，选择最后一行
def vec_filter():
    """
    select the last line for each same name
    :return:
    """
    filepath = 'example_results/vec-result3.txt'
    f = open(filepath, 'r')
    result = []
    for line in f.readlines():
        line = line.split('.sol')
        result.append(line)
    for i in range(len(result)):
        if result[i][0] != result[i - 1][0]:
            open('example_results/vec-update-result3.txt', 'a').write('%s' % ''.join(result[i - 1]))


# 将转化为向量的合约数据集写入txt文件
def catch_file():
    for i in range(9000, 10000):
        filepath = 'contract/contract10/' + str(i) + '.sol'
        vec = word_to_vec(filepath)
        open('featureData/f1/id.txt.feature.txt', 'a').write(
            '%s' % ''.join(str(i) + '.sol' + ' ' + str(vec)) + '\n')  # 保存入结果文件
        print(filepath + " has already done")


# 获取目录下的全部文件名，并写入文件中
def get_filedirname(name):
    dir = os.listdir(name)
    print(dir)
    dirs = dir.sort(key=lambda x: x[1])
    print(dirs)
    fopen = open('featureData/f1/id.txt.txt', 'w')
    for d in dir:
        filename = d + ' '
        fopen.write(filename)
    fopen.close()


# 输入文件名
def input_filename():
    fopen = open('featureData/f4/id.txt.txt', 'w')
    for i in range(40000, 40400):
        filename = str(i) + '.sol' + ' '
        fopen.write(filename)
    fopen.close()


if __name__ == '__main__':
    vul_type_to_vec()
    vec_filter()
    # read_vul_type()
    # catch_file()
    # get_filedirname(fileDirName)
    # input_filename()
    # update_vec()
