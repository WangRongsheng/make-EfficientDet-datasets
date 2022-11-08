import xml.dom.minidom as xmldom
import os

#voc数据集获取所有标签的所有类别数"
annotation_path=r'./yourdatasets/Annotations/'

annotation_names=[os.path.join(annotation_path,i) for i in os.listdir(annotation_path)]

labels = []
for names in annotation_names:
    xmlfilepath = names
    domobj = xmldom.parse(xmlfilepath)
    elementobj = domobj.documentElement
    subElementObj = elementobj.getElementsByTagName("object")
    for s in subElementObj:
        label=s.getElementsByTagName("name")[0].firstChild.data
        if label not in labels:
            labels.append(label)
print(labels)