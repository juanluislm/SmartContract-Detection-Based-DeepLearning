#!/usr/bin/env python
# coding:utf8

import os
import re
import shutil

regtxt = r'.+?\.sol'  # 扫描对象为.sol文件.
regcontent = r'call.value'  # 列出内容含有'call.value'的文件


class FileException(Exception):
    pass


def getDirList(filepath):
    """获取目录下所有的文件."""
    txtlist = []  # 文件集合.
    txtre = re.compile(regtxt)
    needfile = []  # 存放结果.
    for parent, listdir, listfile in os.walk(filepath):
        for files in listfile:
            # 获取所有文件.
            istxt = re.findall(txtre, files)
            filecontext = os.path.join(parent, files)
            # 获取非空的文件.
            if istxt:
                txtlist.append(filecontext)
                # 将所有的数据存放到needfile中.
                needfile.append(readFile(filecontext))

    if needfile == []:
        raise FileException("no file can be find!")
    else:
        validatedata = getValidData(needfile)
        print(validatedata)  # 打印获取的文件名

        # 循环复制数组中的文件
        for i in range(len(validatedata)):
            copyFile(validatedata[i])

        print('total file %s , validate file %s.' % (len(txtlist), len(validatedata)))


def getValidData(filelist=[]):
    """过滤集合中空的元素."""
    valifile = []
    for fp in filelist:
        if fp != None:
            valifile.append(fp)
    return valifile


def readFile(filepath):
    """通过正则匹配文本中内容，并返回文本."""
    contentre = re.compile(regcontent)
    fp = open(filepath)
    lines = fp.readlines()  # 逐行读取文件内容
    flines = len(lines)
    # 逐行匹配数据.
    for i in range(flines):
        iscontent = re.findall(contentre, lines[i])
        if iscontent:
            fp.close()
            return filepath


def copyFile(src_file):
    dst = "/media/jion1/data/train_data"  # 指定文件夹
    shutil.copy(src_file, dst)  # shutil.copy 复制文件到指定文件夹下


if __name__ == "__main__":
    for i in range(41):
        getDirList("/media/jion1/data/contract/contract" + str(i+1) + "/")  # 修改为自己主机上的文件地址
