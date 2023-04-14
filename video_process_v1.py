import cv2
import os
from os import walk, rename, makedirs
from os.path import join, abspath
import time

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



def video2pics0(path_vedio,root_output):
    vc = cv2.VideoCapture(path_vedio)
    rval, frame = vc.read()
    # print(rval,frame.shape,type(frame))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plt.imshow(frame)
    # plt.show()

    fps = vc.get(cv2.CAP_PROP_FPS)

    frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    print("[INFO] 视频时长: {}s".format(frame_all / fps))


    frame_interval=1 #保存帧数间隔
    frame_count = 1
    output_path=os.path.sep.join([root_output,'jpgs_{}'.format(int(time.time()))])
    os.mkdir(output_path)

    filename = os.path.sep.join([output_path, "test_00000.jpg"])
    cv2.imwrite(filename, frame)
    count = 0
    while rval:
        rval, frame = vc.read()
        if frame_count % frame_interval == 0 and frame is not None:
            filename = os.path.sep.join([output_path, "test_{:0>5}.jpg".format(frame_count)])
            cv2.imwrite(filename, frame)
            count += 1
            print("保存图片:{}".format(filename))
        frame_count += 1

    # 关闭视频文件
    vc.release()
    print("[INFO] 总共保存：{}张图片".format(count))

    return output_path

def extract_frames(video_path, dst_folder, index):
    EXTRACT_FREQUENCY=180
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % EXTRACT_FREQUENCY == 0:
            save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
    video.release()
    # 打印出所提取帧的总数
    print("Totally save {:d} pics".format(index-1))


def jpgs2video(path_jpgs,path_output=None):
    timenow=int(time.time())
    path_output=os.path.sep.join([path_jpgs,'cpnverted_vedio_time_{}'.format(timenow)])
    os.mkdir(path_output)


    size=(1920,1080)
    fps=30
    path_output=os.path.join(path_output,'vedio_time_{}.mp4'.format(timenow))
    vw = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    paths = get_path1(path_jpgs)[3]
    for y in paths:
        img = cv2.imread(y)
        vw.write(img)
        print('pic writed :',y)


    vw.release()
    #cv2.destroyAllWindows()
    print('视频合成生成已完成')

    return



def main():
    path_video=r'F:\DL\u2net_02\video_workspace\fishcuts\fish_cut1.mp4'
    path_outpics=r'F:\DL\u2net_02\video_workspace\video_pics'
    #video2pics0(path_video,path_outpics)
    path_markedpics=r'F:\DL\u2net_02\video_workspace\marked_pics'
    jpgs2video(path_markedpics)

    pass

if __name__ == '__main__':
    main()