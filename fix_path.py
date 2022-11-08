import xml.dom.minidom
import os

path = r'./dataset/Annotations'  # xml文件存放路径
sv_path = r'./dataset/Annotations1'  # 修改后的xml文件存放路径
files = os.listdir(path)
cnt = 1

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))
    root = dom.documentElement
    item = root.getElementsByTagName('path')
    for i in item:
        #i.firstChild.data = 'dataset/JPEGImages/' + str(cnt).zfill(6) + '.jpg'  # xml文件对应的图片路径
        i.firstChild.data = str(cnt).zfill(6) + '.jpg'
    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
    cnt += 1