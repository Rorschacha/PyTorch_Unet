import xml.etree.ElementTree as ET
from os import walk, rename, makedirs, getcwd
from os.path import join, abspath
from os.path import split as osplit
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import time
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import sample
import random
import scipy.io as sio


class Count_Xml(object):

    def __init__(self):
        self.set_class = set()
        self.dict_countcls = dict()
        self.dict_countdif = {0: 0, 1: 1}
        self.list_countsize = list()

        # 一些子标签统计
        # self.dict_countnode=dict()
        pattern_seq = dict()
        for x1 in range(2):
            for x2 in range(2):
                for x3 in range(2):
                    for x4 in range(2):
                        for x5 in range(2):
                            for x6 in range(2):
                                pattern_seq['{}{}{}{}{}{}'.format(x1, x2, x3, x4, x5, x6)] = 0
        self.dict_countnode = pattern_seq

    def count_classes(self, cls_name):
        self.set_class.add(cls_name)
        if cls_name in self.dict_countcls.keys():
            self.dict_countcls[cls_name] += 1
        else:
            self.dict_countcls[cls_name] = 1

    def count_sizes(self, size):
        self.list_countsize.append(size)

    def count_difficult(self, difficult):
        if int(difficult) == 0:
            self.dict_countdif[0] += 1
        else:
            self.dict_countdif[1] += 1

    def count_nodes(self, seq):
        if seq in self.dict_countnode.keys():
            self.dict_countnode[seq] += 1

    def gether_infor(self):
        classes = self.set_class
        count_classes = self.dict_countcls
        count_size = ''
        count_difficult = self.dict_countdif

        return classes, count_classes, count_difficult

    def gether_node(self):
        return self.dict_countnode

    def draw_size(self):
        # print('called')
        array_H = []
        array_W = []
        for y in self.list_countsize:
            _, h, w = y
            array_H.append(h)
            array_W.append(w)
            # print('H*W',h,w)

        pd_H = pd.Series(array_H)
        print(pd_H.describe())

        plt.scatter(array_H, array_W)
        plt.show()



import xml.etree.ElementTree as ET
from os import walk, rename, makedirs, getcwd
from os.path import join, abspath
from matplotlib import pyplot as plt
import pandas as pd


def inspect_annotation1(path):
    '''查看正常voc数据集'''
    # 1 取出目标dir下所有xml文件的路径
    dirpath, dirnames, filenames = next(walk(path))
    filepaths = [] #xml路径列表
    for y in filenames:
        if y.endswith('xml'):
            filepaths.append(join(dirpath, y))

    print('  xml root path :', path)
    print('xml files number:', len(filepaths))
    # print(filenames,filepaths)


    for xml_path in filepaths:
        xml_file = open(xml_path)
        tree = ET.parse(xml_file)
        tree_root = tree.getroot()

        # fdesc(tree_root)


        for obj in tree_root.iter('object'):
            # print(obj)
            cls = obj.find('name').text
            #c.count_classes(cls)

