import json
from os import walk, rename, makedirs
from os.path import join, abspath
import datetime

# import time
# import random

#utilisv2 v 0.2.1.1830

def read_json0(path):
    with open(path) as f_obj:
        j_obj = json.load(f_obj)
    return j_obj


def read_txt0(path):
    with open(path, 'r') as f_obj:
        lines = f_obj.readlines()
    return lines


# 工具
def save2json0(obj, path):
    with open(path, 'w') as f_obj:
        json.dump(obj, f_obj)

def write2txt0(obj,path):
    with open(path,'a') as fobj:
        for y in obj:
            fobj.write(y+'\n')

def write2txt1(lines,path):
    with open(path, 'a') as fobj:
        fobj.writelines(lines)


class Log(object):
    def __init__(self, path='log0.txt'):
        self.path = path

    def log0(self, line):
        with open(self.path, 'a') as fobj:
            fobj.write(line + '\n')


# 路径
def get_path1(root_path,show=True):
    '''返回路径下的文件名、文件路径等'''
    dirpath, dirnames, filenames = next(walk(root_path))

    filepaths = []
    for y in filenames:
        filepaths.append(join(dirpath, y))

    if show:
        print('  dirpath :', dirpath)
        print(' dirnames :', dirnames)
        print('filenames :', filenames)
        print('filepaths :', filepaths)
    return dirpath, dirnames, filenames, filepaths

def get_path2(root_path,show=True):
    '''返回路径下的文件名、文件路径等'''
    dirpath, dirnames, filenames = next(walk(root_path))
    filepaths = []
    for y in filenames:
        filepaths.append(join(dirpath, y))
    if show:
        print('  dirpath :', dirpath)
        print(' dirnames :', dirnames)
        print('filenames :', filenames)
        print('filepaths :', filepaths)
    dict_result=dict()
    dict_result['dirpath']=dirpath
    dict_result['dirnames'] =dirnames
    dict_result['filenames']=filenames
    dict_result['filepaths']=filepaths

    return dict_result


#时间
def read_utctime0(timestamp):
    t_utc=datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
    return t_utc

# 其他工具
def name_insert(filename, words):
    '''文件名加入后缀'''
    names = filename.split('.')
    new_name = names[0] + '_' + words + '.' + names[1]
    return new_name


def desc(obj):
    print('----type:', type(obj))
    print('--length:', len(obj))


def fdesc(obj):
    for index, y in enumerate(obj):
        print('-index-', index, '-content-', y)


def main0():
    path0 = r'rowpics/train.txt'
    lines = read_txt0(path0)
    print(lines)
    fdesc(lines)


if __name__ == '__main__':
    main0()
