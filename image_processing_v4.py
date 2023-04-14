# image_process v 0.4.211122
import json
# import shutil
import os
import re
from time import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from time import sleep
import random
import tqdm
from collections import Counter

from skimage import morphology
import math
import datetime

# from utilsV2 import save2json0, read_json0, get_path1, name_insert, get_path2, write2txt1

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存


# utils
# 处理路径
def get_path0(root_path):
    '''check path only'''
    for dirpath, dirnames, filenames in os.walk(root_path):
        print('dirpath', dirpath)
        print('dirnames', dirnames)
        print('filenames', filenames)


def save2json0(obj, path):
    with open(path, 'w') as f_obj:
        json.dump(obj, f_obj)


def rename_pics0(root_path):
    '''目录下的图片重新从0开始命名'''
    dirpath, dirnames, filenames = next(os.walk(root_path))
    old_filepaths = []
    for y in filenames:
        old_filepaths.append(os.path.join(dirpath, y))

    list_new_filenames = []
    numbers = len(filenames)
    for index, y in enumerate(filenames):
        filename_extension = re.split('\.', y)[1]
        new_filename = '{:06}.'.format(index) + filename_extension
        list_new_filenames.append(new_filename)

    new_filepaths = []
    for y in list_new_filenames:
        new_filepaths.append(os.path.join(dirpath, y))

    for index in range(numbers):
        os.rename(old_filepaths[index], new_filepaths[index])

    print('renamed files number:', len(new_filepaths))


# 查看图片
# 直接查看
def show_img(image):
    plt.imshow(image)
    plt.show()


def show_img_cv2(image_array):
    '''cv2版直接查看图像'''
    cv2.imshow("inspect image", image_array)
    cv2.waitKey(0)


# 查看图像
def inspect_img0(**kwargs):
    '''path=...,image=...,open-cv版'''
    path = kwargs.get('path')
    image = kwargs.get('image')
    if path is not None:
        path = kwargs['path']
        image = cv2.imread(path)
    elif image is not None:
        image = kwargs['image']

    if image is not None:
        print('    size :', image.size)
        print('   shape :', image.shape)  # height，width
        print('data type:', image.dtype)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr转换为RGB
        plt.imshow(image)
        plt.show()

    return image


def inspect_img1(**kwargs):
    '''path=...,image=...,pillow版,接受bgr图片/数组 并 强制转换为RGB pillow图片对象'''
    path = kwargs.get('path')
    image = kwargs.get('image')
    show = kwargs.get('show')
    if path is not None:
        path = kwargs['path']
        image = Image.open(path)
    elif image is not None:
        image = kwargs['image']

    if image is not None:
        image = image.convert('RGB')
        print('   size :', image.size)  # height，width
        print(' format :', image.format)
        print('   mode :', image.mode)
        if show:
            plt.imshow(image)
            plt.show()

    return image


def inspect_img2(**kwargs):
    '''path=...,image=...,pillow版,默认获得RBG pillow图像'''
    path = kwargs.get('path')
    image = kwargs.get('image')
    if path is not None:
        path = kwargs['path']
        image = Image.open(path)
    elif image is not None:
        image = kwargs['image']

    if image is not None:
        # image = image.convert('RGB')
        print('   size :', image.size)  # height，width
        print(' format :', image.format)
        print('   mode :', image.mode)
        plt.imshow(image)
        plt.show()

    return image


# 灰度图转换
def rgb2gray():
    '''rbg数组转为单通道灰度图数组'''
    img = cv2.imread("./test.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


pass


def inspect_grayimg0(show=None, **kwargs):
    '''读取三通道，得到单通道图像，opencv'''
    '''path=...,image=...'''
    path = kwargs.get('path')
    image = kwargs.get('image')
    if path is not None:
        path = kwargs['path']
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif image is not None:
        image = kwargs['image']

    if show is not None:
        plt.imshow(image, cmap="gray")
        plt.show()

    if image is not None:
        print('    size :', image.size)
        print('   shape :', image.shape)  # height，width
        print('data type:', image.dtype)
        return image


# advance processing
def gray2binary(gray_array='', threshold=127, show=None):
    '''灰度图二值化'''
    ret, binary_image = cv2.threshold(gray_array, threshold, 255, cv2.THRESH_BINARY)

    if show is not None:
        plt.imshow(binary_image, cmap="gray")
        plt.show()

    return binary_image
    pass


def get_contours(binary_array='', show=None):
    '''获取轮廓点集，需要输入二值化图像'''
    contours, hierarchy = cv2.findContours(binary_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if show is not None:
        # 绘制轮廓线，向内填充
        cv2.drawContours(binary_array, contours, -1, (125, 0, 0), 3)  # 若为单通道图像 则颜色只有第一位生效
        show_img_cv2(binary_array)

    return contours


def get_centroid(single_contour):
    '''获取形心，需要输入单个轮廓点集,返回质心坐标'''
    moments_dict = cv2.moments(single_contour)
    m00 = moments_dict['m00']
    m10 = moments_dict['m10']
    m01 = moments_dict['m01']
    cx = int(m10 / m00)
    cy = int(m01 / m00)
    return cx, cy
    pass


# 保存图像
def save2jpg0(image, path):
    '''pillow版'''
    if path.endswith('jpg'):
        image.save(path)
        print('[pic saved]pic path =', path)
    else:
        raise Exception('[wrong save type]must save jpg pics!')


def save2jpg1(image, path):
    '''opencv版'''
    if path.endswith('jpg'):
        cv2.imwrite(path, image)
        print('[pic saved]pic path =', path)
    else:
        raise Exception('[wrong save type]must save jpg pics!')


def save2png1(image, path):
    '''opencv版'''
    if path.endswith('png'):
        cv2.imwrite(path, image)
        print('[pic saved]pic path =', path)
    else:
        raise Exception('[wrong save type]must save png pics!')


# 绘图
class DrawOperator(object):
    def __init__(self):
        pass

    # 绘图
    # cv2绘图
    def mark_circleB1(self, img, coord, size=20, color=(0, 0, 0), thickness=1):
        '''
        在图像（包括二值图像）上 绘制标记圆,如果是二值图像，color参数只有第一个生效
        :param img: 
        :param coord: (x,y) 
        :return: 
        '''''
        x, y = coord
        cv2.circle(img, (x, y), size, color, thickness)

    def arrow0(self, img, coord):
        '''绘制向下箭头'''
        y, x = coord
        cv2.arrowedLine(img, (x, y - 40), (x, y), 255, 2, 8, 0, 0.3)

    def arrow1(self, img, coord):
        '''绘制向上箭头'''
        y, x = coord
        cv2.arrowedLine(img, (x, y + 40), (x, y), 255, 2, 8, 0, 0.3)

    def arrow2(self, img, coord):
        '''绘制向左箭头'''
        y, x = coord
        cv2.arrowedLine(img, (x - 40, y), (x, y), 255, 2, 8, 0, 0.3)

    def arrow3(self, img, coord):
        '''绘制向右箭头'''
        y, x = coord
        cv2.arrowedLine(img, (x + 40, y), (x, y), 255, 2, 8, 0, 0.3)


# 复合操作
class SeriesOperator(object):
    def __init__(self):
        self.do = DrawOperator()
        pass

    def get_centroid_aio(self, path=r''):
        '''put in a png image path, return (cx, cy)'''
        path_gray = path
        binary_threshold = 200

        gray_array = cv2.imread(path_gray, cv2.IMREAD_GRAYSCALE)
        binary_img = gray2binary(gray_array, binary_threshold, None)

        contours = get_contours(binary_img, None)
        for _contour in contours:
            if len(_contour) > 500:
                contour = _contour
                break
        # moments_dict = cv2.moments(contour)
        # contour=contours[]

        cx, cy = get_centroid(contour)
        print('get centroid of {},which is ({},{})'.format(path_gray, cx, cy))

        # 可视化
        if False:
            self.do.mark_circleB1(binary_img, (cx, cy), 50, (100, 0, 0), 5)
            # show_img(binary_img)
            # inspect_grayimg0(binary_img)
            # cv2.circle(binary_img,(cx,cy), 10, (0, 0, 255),5)
            show_img_cv2(binary_img)
            pass

        return cx, cy




def main():
    do = DrawOperator()
    so = SeriesOperator()
    #
    path_gray = r'F:\DL\u2net_02\test_outputs\test_00182.jpg'
    gray_array = inspect_grayimg0(path=path_gray, show=True)

    if False:
        path_gray = r'F:\DL\u2net_02\test_outputs\test_00232.jpg'
        gray_array = inspect_grayimg0(path=path_gray, show=True)
        binary_img = gray2binary(gray_array, 200, True)

        contours = get_contours(binary_img, True)
        contour = contours[0]
        # moments_dict = cv2.moments(contour)
        # contour=contours[]

        cx, cy = get_centroid(contour)
        print('check')
        if True:
            do.mark_circleB1(binary_img, (cx, cy), 50, (100, 0, 0), 5)
            # show_img(binary_img)
            # inspect_grayimg0(binary_img)
            # cv2.circle(binary_img,(cx,cy), 10, (0, 0, 255),5)
            show_img_cv2(binary_img)

    #cx, cy = so.get_centroid_aio(path_gray)

    print('check')
    # cx,cy=so.get_centroid_aio(path_gray)

    '''--json_dict:

    ---data_centroid :(index_id,path, (cx,cy) )   0~n
    option key-value：
    ---data_rotation_angle:(index_id,path, (cx,cy),   (gama,theta ) )     1~n,index从1开始
    theta-t=arcsin[ (y(t)-y(t-1)) / gama-t ]
    gama-t=sqrt( (x(t)-x(t-1))^2+(y(t)-y(t-1))^2     )
    ---data_v:(index_id,path, (cx,cy),   deltaT,(dx,dy),(vx,vy) )     1~n,index从1开始
    
    ---stat : dict( vmax:,vmin:, theta-max:,theta-min:)
    ---description：(time,centroid numbers)
    
    '''

    png_dir = r'F:\DL\u2net_02\test_outputs'

    root_path = png_dir
    dirpath, dirnames, filenames = next(os.walk(root_path))
    filepaths = []
    for filename in filenames:
        if filename.endswith('jpg'):
            filepaths.append(os.path.join(dirpath, filename))

    data_centroid = []
    data_rotation_angle = []
    data_v = []

    for index_id, image_path in enumerate(filepaths):
        cx, cy = so.get_centroid_aio(image_path)
        # build basic centroid data
        if True:  # ---data_centroid :(index_id,path, (cx,cy) )   0~n
            write_in = (index_id, image_path, (cx, cy))
            data_centroid.append(write_in)

    # build option data

    if True:  # data_rotation_angle
        deltaFrame = 4
        deltaT = deltaFrame * 1
        for basic_write_in in data_centroid:
            index_id, image_path, centroid_coord = basic_write_in
            cx, cy = centroid_coord
            if index_id == 0:
                cxt0, cyt0 = cx, cy
            elif index_id > 0 and index_id % deltaFrame == 0:
                gama = math.sqrt((cx - cxt0) ** 2 + (cy - cyt0) ** 2)
                if gama != 0:
                    deltay = cy - cyt0
                    theta = math.asin(deltay / gama)
                    write_in_ra = (index_id, image_path, (cx, cy), (gama, theta))
                    data_rotation_angle.append(write_in_ra)
                else:
                    print('centroid did not move')
                    write_in_ra = (index_id, image_path, (cx, cy), (0, 0))
                    data_rotation_angle.append(write_in_ra)
                    pass

                cxt0, cyt0 = cx, cy  # log cx(t-1),cy(t-1)

    if True:  # data_v
        deltaFrame = 4
        deltaT = deltaFrame * 1
        for basic_write_in in data_centroid:
            index_id, image_path, centroid_coord = basic_write_in
            cx, cy = centroid_coord
            if index_id == 0:
                cxt0, cyt0 = cx, cy
            elif index_id > 0 and index_id % deltaFrame == 0:
                dx = (cxt0 + cx) / 2
                dy = (cyt0 + cy) / 2
                vx = (cx - cxt0) / deltaT
                vy = (cx - cxt0) / deltaT
                write_in_v = (index_id, image_path, (cx, cy), deltaT, (dx, dy), (vx, vy))
                data_v.append(write_in_v)

                cxt0, cyt0 = cx, cy  # log cx(t-1),cy(t-1)

    # write2json
    json_data = {
        'data_centroid': data_centroid,
        'data_rotation_angle': data_rotation_angle,
        'data_v': data_v
    }


    t_mark = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    json_output_dir = r'F:\DL\u2net_02\test_outputs_logs\workspace'
    json_path = os.path.join(json_output_dir, 'sequence_infor{}.json'.format(t_mark))
    #save2json0(json_path,json_data)
    with open(json_path, 'w') as f_obj:
        json.dump(json.dumps(json_data), f_obj)

    print('end')
    pass


if __name__ == '__main__':
    main()
