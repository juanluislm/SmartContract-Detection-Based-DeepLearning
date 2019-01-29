import os

path = r'./dataset'
dir = os.listdir(path)
fopen = open('./featureData/f1/text.txt', 'w')

for d in dir:            # d是每一个文件的文件名
    string = d + ' '     # 拼接字符串并换行
    fopen.write(string + '\n')  # 写入文件中
fopen.close()
