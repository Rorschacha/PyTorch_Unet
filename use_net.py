import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
import glob
import argparse

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB
import time


#
def inspect_npy(path=r''):
    array_read=np.load(path)
    print(array_read)
    print("load .npy done")
    return array_read


def show_img(image):
    plt.imshow(image)
    plt.show()


# normalize the predicted SOD probability map
def my_collate(batch):
    batch = list(filter(lambda img: img is not None, batch))
    return torch.utils.data.dataloader.default_collate(list(batch))


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    try:
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        post_process_time = time.time()
        im = Image.fromarray(predict_np * 255).convert('RGB')
        img_name = image_name.split("/")[-1]
        image = io.imread(image_name)
        imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

        pb_np = np.array(imo)
        image_filter = np.greater(pb_np, 200)
        only_image_name = img_name.split("/")[-1].split(".")[0]
        output_path = os.path.join(d_dir, only_image_name)
        save_time = time.time()
        np.save(output_path, image_filter)
        # aaa = img_name.split(".")
        # bbb = aaa[0:-1]
        # imidx = bbb[0]
        # for i in range(1, len(bbb)):
        #     imidx = imidx + "." + bbb[i]
        #
        # imo.save(d_dir + imidx + '.png')
    except Exception as error:
        raise Exception(error)

def modified_save_output(image_name, pred, d_dir):
    try:
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        post_process_time = time.time()
        im = Image.fromarray(predict_np * 255).convert('RGB')
        image = io.imread(image_name)
        imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

        pb_np = np.array(imo)
        #image_filter = np.greater(pb_np, 200)
        #filename_without_extension, extension = os.path.splitext(image_name)
        folder_path, filename_with_extension = os.path.split(image_name)
        #only_image_name=filename_without_extension
        output_path = os.path.join(d_dir, filename_with_extension)


        output_img=Image.fromarray(pb_np)
        #show
        if False:
            plt.imshow(output_img)
            plt.show()

        #save
        if True:
            print(output_path)
            output_img.save(output_path)
            #np.save(output_path, image_filter)
            pass



    except Exception as error:
        raise Exception(error)

def get_parameters():
    parser = argparse.ArgumentParser(
        description="Identifying Salient Object Detection")
    parser.add_argument("-i",
                        "--input",
                        help="Path to the file that lists all path to images",
                        type=str)
    parser.add_argument("-o",
                        "--output_dir",
                        help="Path to the output dir", type=str)

    parser.add_argument("-e",
                        "--errorFile",
                        help="Path to the log error file", type=str)
    args = parser.parse_args()

    return args


def use_net():
    model_path=r'F:\DL\u2net_02\saved_models\u2net\backup\u2net_bce_itr_300_train_2.011912_tar_0.300187.pth'
    test_dir=r'F:\DL\u2net_02\test_images'

    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp


    fully_trained_model=r'F:\DL\u2net_02\attachments\u2net.pth'
    test_trained_model=r'F:\DL\u2net_02\saved_models\u2net\backup\u2net_bce_itr_300_train_2.011912_tar_0.300187.pth'
    fish_model=r'F:\DL\u2net_02\saved_models\u2net\u2net_bce_itr_2200_train_0.247946_tar_0.030409.pth'
    model_path=fully_trained_model

    test_dir=r'F:\DL\u2net_02\test_fish'
    wait4mark_dir=r'F:\DL\u2net_02\video_workspace\video_pics\jpgs_1637205313'

    prediction_dir=r'F:\DL\u2net_02\test_outputs'
    error_file_link=r'F:\DL\u2net_02\test_errorlog'


    model_dir=model_path
    root_path=wait4mark_dir #input dir
    dirpath, dirnames, filenames = next(os.walk(root_path))
    filepaths = [] #路径列表
    for filename in filenames:
        if filename.endswith('jpg'):
            filepaths.append(os.path.join(dirpath, filename))

    print(' root path :', root_path)
    print('files number:', len(filepaths))
    img_name_list=filepaths

    print("Num of image paths in ", str(test_dir), "is: ", len(img_name_list))

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )

    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        collate_fn=my_collate)

    # --------- 3. model define ---------
    if model_name == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        try:
            print("\r------In processing file {} with name {}--------".format(i_test + 1, img_name_list[i_test].split("/")[-1]), end='')

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)


            # normalization
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            # save results to test_results folder
            #save_output(img_name_list[i_test], pred, prediction_dir)
            modified_save_output(img_name_list[i_test], pred, prediction_dir)


            del d1, d2, d3, d4, d5, d6, d7
        except Exception as error:
            print(error)
            with open(error_file_link, 'a+') as err_file:
                error_mess = img_name_list[i_test] + '*' + str(error) + '\n'
                err_file.write(error_mess)
            continue

def main():
    use_net()

    if False:
        npy_path=r'F:\DL\u2net_02\test_images\ILSVRC2012_test_00000082.npy'
        arr=inspect_npy(npy_path)
        show_img(arr)

    pass


if __name__ == "__main__":
    main()

