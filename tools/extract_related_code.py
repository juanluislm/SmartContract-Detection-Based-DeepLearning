import os


# 定位文件中call.value的位置
def find_location(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()
    line_len = len(lines)
    location = 0
    for i in range(line_len):
        if 'call.value' in lines[i]:
            location = i
    out_selected_code(filepath, location)


# output selected code
def out_selected_code(filepath, location):
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()
    line_len = len(lines)
    result = None
    result1 = None
    result2 = None
    if location - 10 >= 0:
        result1 = lines[location - 10:location]
        print('前10行：', result1)
    else:
        result1 = lines[0:location]
        print('前10行：', result1)
    if location + 10 <= line_len:
        result2 = lines[location:location + 10]
        print('后10行：', result2)
    else:
        result2 = lines[location:line_len]
        print('后10行：', result2)
    result = result1 + result2
    print(result)
    newFilePath = '../train_data_V2/'
    writeResult(newFilePath + filepath.split('/')[2], result)


# write result back to file
def writeResult(newfilepath, lines):
    f = open(newfilepath, "w")
    for line in lines:
        f.write(line)
    f.close()


if __name__ == "__main__":
    dirs = os.listdir("../train_data_V1")
    print(len(dirs))
    for file in dirs:
        find_location('../train_data_V1/' + file)



