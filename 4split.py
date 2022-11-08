import os
import random
import sys


root_path = './yourdatasets'
xmlfilepath = root_path + '/Annotations'
txtsavepath = './ImageSets/Main'


if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

train_test_percent = 1.0  # (训练集+验证集)/(训练集+验证集+测试集)
train_valid_percent = 0.9  # 训练集/(训练集+验证集)

total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * train_test_percent)
ts = int(num-tv) 
tr = int(tv * train_valid_percent)
tz = int(tv-tr)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and valid size:", tv)
print("train size:", tr)
print("test size:", ts)
print("valid size:", tz)

# ftrainall = open(txtsavepath + '/ftrainall.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fvalid = open(txtsavepath + '/valid.txt', 'w')

ftestimg = open(txtsavepath + '/img_test.txt', 'w')
ftrainimg = open(txtsavepath + '/img_train.txt', 'w')
fvalidimg = open(txtsavepath + '/img_valid.txt', 'w')

ftest_no = open(txtsavepath + '/test_no.txt', 'w')
ftrain_no = open(txtsavepath + '/train_no.txt', 'w')
fvalid_no = open(txtsavepath + '/valid_no.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '.xml' + '\n'
    # 非jpg记得修改
    imgname = total_xml[i][:-4] + '.jpg' + '\n'
    
    name_no = total_xml[i][:-4] + '\n'
    # 非jpg记得修改
    imgname_no = total_xml[i][:-4] + '\n'
    
    if i in trainval:
        if i in train:
            ftrain.write(name)
            ftrainimg.write(imgname)
            ftrain_no.write(name_no)
        else:
            fvalid.write(name)
            fvalidimg.write(imgname)
            fvalid_no.write(name_no)
    else:
        ftest.write(name)
        ftestimg.write(imgname)
        ftest_no.write(name_no)

ftrain.close()
fvalid.close()
ftest.close()

ftrainimg.close()
fvalidimg.close()
ftestimg.close()

ftest_no.close()
ftrain_no.close()
fvalid_no.close()