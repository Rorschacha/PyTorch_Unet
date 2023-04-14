import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import visdom
from skimage import io, transform
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP




def inspect_img2(**kwargs):
    '''path=...,image=...,pillow版,不转换RBG'''
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


def preprocess_img(image_name,outs):
    predict = outs
    predict = predict.squeeze()
    #predict_np = predict.cpu().data.numpy()
    predict_np = predict.data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo) #HWC
    pb_np_chw=pb_np.transpose((2,0,1)) #CHW

    #output_img=Image.fromarray(pb_np)

    return pb_np_chw


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    '''bce loss'''
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))

    return loss0, loss



def raw_train():

    # misc setup
    model_name = 'u2net' #'u2netp'

    data_dir = './train_data/'
    tra_image_dir = 'DUTS/DUTS-TR/DUTS-TR/im_aug/'
    tra_label_dir = 'DUTS/DUTS-TR/DUTS-TR/gt_aug/'

    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = './saved_models/' + model_name +'/'

    epoch_num = 100000
    batch_size_train = 12
    batch_size_val = 1
    train_num = 0
    val_num = 0


    # new dir
    data_dir = r'F:\datasets'
    tra_image_dir = r'\DUTS\DUTS-TR\DUTS-TR-Image'
    tra_label_dir = r'\DUTS\DUTS-TR\DUTS-TR-Mask'

    #pathname_rule=data_dir + tra_image_dir + '\*' + image_ext
    # 'F:\\datasetsDUTS/DUTS-TR/DUTS-TR/im_aug/*.jpg'
    # tra_img_name_list = glob.glob(pathname_rule)
    root_path=data_dir+tra_image_dir
    dirpath, dirnames, filenames = next(os.walk(root_path))
    filepaths = [] #路径列表
    for filename in filenames:
        if filename.endswith('jpg'):
            filepaths.append(os.path.join(dirpath, filename))

    print(' root path :', root_path)
    print('files number:', len(filepaths))

    tra_img_name_list=filepaths

    tra_lbl_name_list = []
    for filename in filenames:
        if filename.endswith('jpg'):
            label_name=filename.replace("jpg",'png')
            label_dir=data_dir+tra_label_dir
            label_path=os.path.join(label_dir, label_name)
            tra_lbl_name_list.append(label_path)



    if True:
        inspect_id=5
        inspect_img2(path=tra_img_name_list[inspect_id])
        inspect_img2(path=tra_lbl_name_list[inspect_id])
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")



    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)



    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000 # save the model every 2000 iterations


    # train main body
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0


def train_with_tiny_datasets():

    # misc setup
    model_name = 'u2net' #'u2netp'

    data_dir = './train_data/'
    tra_image_dir = 'DUTS/DUTS-TR/DUTS-TR/im_aug/'
    tra_label_dir = 'DUTS/DUTS-TR/DUTS-TR/gt_aug/'

    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = './saved_models/' + model_name +'/'

    epoch_num = 10
    batch_size_train = 12
    batch_size_val = 1
    train_num = 0
    val_num = 0


    # new dir
    data_dir = r'F:\datasets'
    tra_image_dir = r'\DUTS\DUTS-TR\tiny\DUTS-TR-Image'
    tra_label_dir = r'\DUTS\DUTS-TR\tiny\DUTS-TR-Mask'

    #pathname_rule=data_dir + tra_image_dir + '\*' + image_ext
    # 'F:\\datasetsDUTS/DUTS-TR/DUTS-TR/im_aug/*.jpg'
    # tra_img_name_list = glob.glob(pathname_rule)
    root_path=data_dir+tra_image_dir
    dirpath, dirnames, filenames = next(os.walk(root_path))
    filepaths = [] #路径列表
    for filename in filenames:
        if filename.endswith('jpg'):
            filepaths.append(os.path.join(dirpath, filename))

    print(' root path :', root_path)
    print('files number:', len(filepaths))

    tra_img_name_list=filepaths

    tra_lbl_name_list = []
    for filename in filenames:
        if filename.endswith('jpg'):
            label_name=filename.replace("jpg",'png')
            label_dir=data_dir+tra_label_dir
            label_path=os.path.join(label_dir, label_name)
            tra_lbl_name_list.append(label_path)



    if True:
        inspect_id=5
        inspect_img2(path=tra_img_name_list[inspect_id])
        inspect_img2(path=tra_lbl_name_list[inspect_id])
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")



    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)



    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 100 # save the model every 2000 iterations


    # train main body
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))




            if ite_num % save_frq == 0:

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

def train_with_visdom():

    # misc setup
    # set up visdom
    viz = visdom.Visdom(env='u2net_训练测试')

    model_name = 'u2net' #'u2netp'

    data_dir = './train_data/'
    tra_image_dir = 'DUTS/DUTS-TR/DUTS-TR/im_aug/'
    tra_label_dir = 'DUTS/DUTS-TR/DUTS-TR/gt_aug/'

    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = './saved_models/' + model_name +'/'

    epoch_num = 10
    batch_size_train = 12
    batch_size_val = 1
    train_num = 0
    val_num = 0


    # new dir
    data_dir = r'F:\datasets'
    tra_image_dir = r'\DUTS\DUTS-TR\tiny\DUTS-TR-Image'
    tra_label_dir = r'\DUTS\DUTS-TR\tiny\DUTS-TR-Mask'

    #pathname_rule=data_dir + tra_image_dir + '\*' + image_ext
    # 'F:\\datasetsDUTS/DUTS-TR/DUTS-TR/im_aug/*.jpg'
    # tra_img_name_list = glob.glob(pathname_rule)
    root_path=data_dir+tra_image_dir
    dirpath, dirnames, filenames = next(os.walk(root_path))
    filepaths = [] #路径列表
    for filename in filenames:
        if filename.endswith('jpg'):
            filepaths.append(os.path.join(dirpath, filename))

    print(' root path :', root_path)
    print('files number:', len(filepaths))

    tra_img_name_list=filepaths

    tra_lbl_name_list = []
    for filename in filenames:
        if filename.endswith('jpg'):
            label_name=filename.replace("jpg",'png')
            label_dir=data_dir+tra_label_dir
            label_path=os.path.join(label_dir, label_name)
            tra_lbl_name_list.append(label_path)



    if True:
        inspect_id=5
        inspect_img2(path=tra_img_name_list[inspect_id])
        inspect_img2(path=tra_lbl_name_list[inspect_id])
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")



    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)



    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 100 # save the model every 2000 iterations


    # train main body
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)

            #check output
            if True:
                if ite_num%10==0:
                    if False:
                        origin_pic_tensor=data['image'][0,:,:,:].clone()
                        viz.image(
                            origin_pic_tensor,win='U2net_origin_pic',
                            opts=dict(title='U2net_origin', caption='U2net_middle_input')
                        )
                        del origin_pic_tensor

                    outs=d0.cpu()
                    pred = outs[0, 0, :, :]
                    pred = normPRED(pred)
                    processed_img=preprocess_img(tra_img_name_list[i], pred)
                    viz.image(
                        processed_img,win='U2net_pic',
                        opts=dict(title='U2net', caption='U2net_middle_output')
                    )

            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()



            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            train_loss=running_loss / ite_num4val
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if True:
                viz.line(
                    X=np.array([ite_num]),
                    Y=np.array([train_loss]),
                    win='loss_log2{}'.format(epoch),
                    name='batch loss',
                    update='append'
                )



            if ite_num % save_frq == 0:

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0


def real_train():

    # misc setup
    # set up visdom
    viz = visdom.Visdom(env='u2net_训练测试')

    model_name = 'u2net' #'u2netp'

    data_dir = './train_data/'
    tra_image_dir = 'DUTS/DUTS-TR/DUTS-TR/im_aug/'
    tra_label_dir = 'DUTS/DUTS-TR/DUTS-TR/gt_aug/'

    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = './saved_models/' + model_name +'/'

    epoch_num = 30
    batch_size_train = 12
    batch_size_val = 1
    train_num = 0
    val_num = 0


    # new dir
    data_dir = r'F:\datasets\fish\Fish_Dataset\Fish_Dataset\Red Sea Bream'
    tra_image_dir = r'\Red Sea Bream'
    tra_label_dir = r'\Red Sea Bream GT'

    #pathname_rule=data_dir + tra_image_dir + '\*' + image_ext
    # 'F:\\datasetsDUTS/DUTS-TR/DUTS-TR/im_aug/*.jpg'
    # tra_img_name_list = glob.glob(pathname_rule)
    root_path=data_dir+tra_image_dir
    dirpath, dirnames, filenames = next(os.walk(root_path))
    filepaths = [] #路径列表
    for filename in filenames:
        if filename.endswith('png'):
            filepaths.append(os.path.join(dirpath, filename))

    print(' root path :', root_path)
    print('files number:', len(filepaths))

    tra_img_name_list=filepaths

    tra_lbl_name_list = []
    for filename in filenames:
        if filename.endswith('png'):
            label_name=filename.replace("jpg",'png')
            label_dir=data_dir+tra_label_dir
            label_path=os.path.join(label_dir, label_name)
            tra_lbl_name_list.append(label_path)



    if True:
        inspect_id=5
        inspect_img2(path=tra_img_name_list[inspect_id])
        inspect_img2(path=tra_lbl_name_list[inspect_id])
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")



    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)



    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 200 # save the model every 2000 iterations


    # train main body
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)

            #check output
            if True:
                if ite_num%10==0:
                    if False:
                        origin_pic_tensor=data['image'][0,:,:,:].clone()
                        viz.image(
                            origin_pic_tensor,win='U2net_origin_pic',
                            opts=dict(title='U2net_origin', caption='U2net_middle_input')
                        )
                        del origin_pic_tensor

                    outs=d0.cpu()
                    pred = outs[0, 0, :, :]
                    pred = normPRED(pred)
                    processed_img=preprocess_img(tra_img_name_list[i], pred)
                    viz.image(
                        processed_img,win='U2net_pic',
                        opts=dict(title='U2net', caption='U2net_middle_output')
                    )

            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()



            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            train_loss=running_loss / ite_num4val
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if True:
                viz.line(
                    X=np.array([ite_num]),
                    Y=np.array([train_loss]),
                    win='loss_log2{}'.format(epoch),
                    name='batch loss',
                    update='append'
                )



            if ite_num % save_frq == 0:

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

def main():
    #raw_train()
    #train_with_tiny_datasets()
    #train_with_visdom()
    real_train()
    pass



if __name__ == "__main__":
    main()