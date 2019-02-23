# coding:utf-8
import os

flag = False


# Handle Comment of sol source code file("//")
# The comment of file will be deleted if exist lines[i]
def Handle_single_comment(lines, i):
    index = lines[i].find("//")
    if index != -1:
        # lines[i] = lines[i][0:index]
        lines[i] = lines[i].replace(lines[i], '', 1)
        lines[i] += ''


# @brief: Handle Comment of sol source code file("/* */")
# @return -1:the Line is Comment Line,should delete this line
# @return -2:Only begin Comment found in this Line
# @return  0:Not find Comment
def Handle_document_comment(lines, i):
    global flag
    while True:
        if not flag:
            index = lines[i].find("/*")
            if index != -1:
                flag = True
                index2 = lines[i].find("*/", index + 2)
                if index2 != -1:
                    lines[i] = lines[i].replace(lines[i], '', 1)
                    flag = False  # continue look for comment
                else:
                    lines[i] = lines[i].replace(lines[i], '', 1)  # only find "begin comment
                    lines[i] += ''
                    return -2
            else:
                return 0  # not find
        else:
            index2 = lines[i].find("*/")
            if index2 != -1:
                flag = False
                lines[i] = lines[i].replace(lines[i], '', 1)  # continue look for comment
            else:
                return -1  # should delete this


# @brief: Remove Comment of file
# At last print the handled result
def RemoveComment(file):
    global flag
    f = open(file, "r")
    lines = f.readlines()
    f.close()
    length = len(lines)
    i = 0
    while i < length:
        ret = Handle_document_comment(lines, i)
        if ret == -1:
            if flag == False:
                print("There must be some wrong")
            del lines[i]
            i -= 1
            length -= 1
        elif ret == 0:
            Handle_single_comment(lines, i)
        else:
            pass
        i += 1

    Output(lines)
    writeResult(file, lines)


# print result
def Output(lines):
    for line in lines:
        if line == '':
            continue
        print(line)


# write result back to file
def writeResult(file, lines):
    f = open(file, "w")
    for line in lines:
        if line == '':
            continue
        f.write(line)
    f.close()


# remove blank space of source file
def deleteBlank(file):
    f = open(file, 'r')
    fnew = open(file + '1.txt', 'w')
    try:
        for line in f.readlines():
            if line == '':
                line = line.strip(" ")
            fnew.write(line)
    finally:
        f.close()
        fnew.close()


if __name__ == '__main__':
    dirs = os.listdir("../train_data")
    print(len(dirs))
    for file in dirs:
        # print("../train_data/" + file)
        RemoveComment("../train_data/" + file)

