# post_processing v 0.1.211124
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
from pylab import mpl
from skimage import morphology
import math
import datetime

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存






class TrajectoryDisplay(object):
    def __init__(self):
        pass


    def read_json(self,path):
        '''read trajectory json data'''
        with open(path) as f_obj:
            j_obj = json.load(f_obj)
        return j_obj


    def draw_displace(self):
        '''空白画纸上绘制形心'''
        json_path=r'F:\DL\u2net_02\test_outputs_logs\workspace\sequence_infor2021-11-23_17-27-20.json'
        json_data=json.loads(self.read_json(json_path))

        data_centroid=json_data['data_centroid']
        data_rotation_angle=json_data['data_rotation_angle']
        data_v=json_data['data_v']


        #位移图
        #(1080, 1920)
        data_length=len(data_centroid)
        empty_array=np.zeros([1080, 1920],dtype=np.uint8)+255
        for loop_index,write_in in enumerate(data_centroid):
            index_id,path,centroid =write_in
            cx,cy=centroid

            color_number=int(151-loop_index/data_length*(150-0))

            cv2.circle(empty_array, (cx,cy), 10, (color_number,0,0), 2)


        cv2.imshow("inspect image", empty_array)
        cv2.waitKey(0)

    def draw_displace_line(self):
        '''形心折线图'''
        pass

    def draw_v(self):
        json_path=r'F:\DL\u2net_02\test_outputs_logs\workspace\sequence_infor2021-11-23_17-27-20.json'
        json_data=json.loads(self.read_json(json_path))

        data_v=json_data['data_v']
        processed_data_v=[]
        for loop_index,write_in in enumerate(data_v):
            index_id,path,centroid,deltaT,v_coord,v_component =write_in
            cx,cy=centroid
            dx,dy=v_coord
            vx,vy=v_component
            v=math.sqrt(vx**2+vy**2)
            processed_data_v.append(v)


        if True:
            plt.hist(processed_data_v, 10 )
            plt.xlabel("速度区间（单位：像素/帧数间隔时间）")
            plt.ylabel("频数")
            plt.title("速度分布直方图")
            plt.show()

        print('end of draw v')


    def draw_r(self):
        json_path=r'F:\DL\u2net_02\test_outputs_logs\workspace\sequence_infor2021-11-23_17-27-20.json'
        json_data=json.loads(self.read_json(json_path))

        data_rotation_angle=json_data['data_rotation_angle']
        processed_data_rotation_angle=[]
        for loop_index,write_in in enumerate(data_rotation_angle):
            index_id,path,centroid,r_coord=write_in
            cx,cy=centroid
            gama,theta=r_coord
            theta_degree=theta/(2*math.pi)*360
            processed_data_rotation_angle.append(theta_degree)


        if True:
            plt.hist(processed_data_rotation_angle, 10 )
            plt.xlabel("转角度数区间")
            plt.ylabel("频数")
            plt.title("转角分布直方图")
            plt.show()

        print('end of draw r')


class SeriesOperator(object):
    def __init__(self):
        pass




def main():
    so=SeriesOperator()
    td=TrajectoryDisplay()
    #位移图
    #td.draw_displace()

    #速度分布
    #td.draw_v()

    #转角分布
    #td.draw_r()

    print('end')



if __name__ == '__main__':
    main()