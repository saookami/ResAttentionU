from __future__ import print_function
# from keras.preprocessing.image import ImageDataGeneratortif
# from keras.preprocessing.image import ImageDataGeneratorgay
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
import sys
import gdal
import skimage
import random
import json
# from osgeo import gdal

Water = [0,36,155]
Road = [249,255,96]
BareLand = [165,224,255]
Vegtation = [154 ,235, 141]
Building = [193,142,100]
OtherImpervious = [123,123,123]

COLOR_DICT = np.array([Vegtation, Building, Road , OtherImpervious , BareLand, Water])



#原始的adjustData
# def adjustData(img,mask,flag_multi_class,num_class):
#     if(flag_multi_class):
#         img = img / 255
#         mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
#         new_mask = np.zeros(mask.shape + (num_class,))
#         for i in range(num_class):
#             #for one pixel in the image, find the class in mask and convert it into one-hot vector
#             #index = np.where(mask == i)
#             #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
#             #new_mask[index_mask] = 1
#             new_mask[mask == i,i] = 1
#         new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
#         mask = new_mask
#     elif(np.max(img) > 1):
#         img = img / 255
#         mask = mask /255
#         mask[mask > 0.5] = 1
#         mask[mask <= 0.5] = 0
#     return (img,mask)
#新的adjustdata https://blog.csdn.net/qq_39622795/article/details/105688706
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):#多类情况
        # print('mask.shape',mask.shape)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # angle = random.randint(1,4)
        # img = skimage.transform.rotate(img, angle=angle, mode='reflect')#旋转
        # mask = skimage.transform.rotate(mask, angle=angle, mode='reflect')#旋转
        # print('mask.shape',mask.shape)(2, 256, 256, 1)

        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        # mask = mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))

        new_mask[mask == 1,0] = 1
        new_mask[mask == 2,1] = 1
        new_mask[mask == 3,2] = 1
        new_mask[mask == 4,3] = 1
        new_mask[mask == 5,4] = 1
        new_mask[mask == 6,5] = 1
        new_mask[mask == 255,0] = 1
        mask = new_mask
        # print(mask.shape,'mask.shape')
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "tif",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 6,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    ''''categorical'
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directorytiff(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        # print(img.shape,'img shape')
        img,mask = adjustData(img,mask,flag_multi_class,num_class)

        yield (img,mask)

def load_tiff_to_array(img_file_path):
    """
    读取栅格数据，将其转换成对应数组
    :param: img_file_path: 栅格数据路径
    :return: 返回投影，几何信息，和转换后的数组(5888,5888,10)
    """
    # print('path', img_file_path)
    dataset = gdal.Open(img_file_path)  # 读取栅格数据
    Bands = dataset.RasterCount
    # print('path',img_file_path,'处理图像的栅格波段数总共有：', dataset.RasterCount)

    # 判断是否读取到数据
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出

    projection = dataset.GetProjection()  # 投影
    transform = dataset.GetGeoTransform()  # 几何信息

    ''' 栅格图片转成数组,依次读取每个波段，然后转换成数组的每个通道 '''

    array_channel = 0  # 数组从通道从0开始
    # print('处理其中的波段数：', Bands)

    # 读取Bands列表中指定的波段
    for band in range(1,Bands+1):  # 图片波段从1开始
        # print('正在将其中的{}波段转换成数组'.format(band))
        srcband = dataset.GetRasterBand(band)  # 获取栅格数据集的波段
        if srcband is None:
            # print('WARN: srcband is None: ' + str(band) + img_file_path)
            continue

        #  一个通道转成一个数组（5888,5888）
        arr = srcband.ReadAsArray()

        if array_channel == 0:  # 如果是第一个通道，要新建数组（5888,5888,10）
            H = arr.shape[0]  # 图像的长 5888
            W = arr.shape[1]  # 图像的宽 5888
            img_array = np.zeros((H, W, Bands), dtype=np.float32)

        img_array[:, :, array_channel] = np.float32(arr)

        array_channel += 1  # 进入下一个波段-> 数组

    return img_array

def testGenerator(test_path,num_image,target_size = (256,256),flag_multi_class = True,as_gray = None):
    f_name = sorted(os.listdir(test_path))
    # f_name.sort(key=lambda x: int(x[4:-5]))
    # f_name.sort(key=lambda x: int(x[:-4]))
    # f_name.sort(key=lambda x: int(x[:-5]))
    number_p = num_image

    # print('number_img',range(number_p))
    for i in range(number_p):
        i_path = os.path.join(test_path,f_name[i])
        # print(i_path)
        img = load_tiff_to_array(i_path)#
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # print('img.shape',img.shape)
        img = trans.resize(img,target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        # print(img.shape)
        yield img

def geneTrainNpy(image_path,mask_path,flag_multi_class = True,num_class = 6,image_prefix = "image",mask_prefix = "mask",image_as_gray = False,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    img3 = np.zeros((256, 256, 50))

    for index,item in enumerate(image_name_arr):
        if image_as_gray != True:
            img1 = gdal.Open(item)
            im_width = img1.RasterXSize  # 栅格矩阵的列数
            im_height = img1.RasterYSize  # 栅格矩阵的行数
            i,j,k = 0,0,0
            for i in range(10):
                band = img1.GetRasterBand(i+1)
                # print(i+1)
                im_datas = band.ReadAsArray(0, 0, im_width, im_height)
                for j in range(512):
                    for k in range(512):
                        img3[j,k,i] = im_datas[j,k]
            img = np.array(img3)
            # print('readtiff')

        else:
            img = io.imread(item)
            # print('read mask')
    img = np.reshape(img3,img3.shape + (1,)) if image_as_gray else img
    mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
    mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
    img,mask = adjustData(img,mask,flag_multi_class,num_class)
    image_arr.append(img)
    mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def labelVisualize(num_class,color_dict,img):
    # print(img.shape,'img')#(256, 256) img
    img2 = img[:,:,0] if len(img.shape) == 3 else img
    img = img * 255.0
    img_out = np.zeros(img2.shape + (3,))
    # print(img_out.shape,'img_out')#(256, 256, 3)
    for i in range(num_class):
        print(img[:,:,i].max(),img[:,:,i].min())
        img_out[img[:,:,i] >= 150] = color_dict[i]
        # img_out[img == i,:] = color_dict[i]
    return img_out
    # return img_out

def saveResult(save_path,npyfile,flag_multi_class = True,num_class = 6):
    f_name = sorted(os.listdir("E:/python/unet-master/data/membrane_47/test50/image"))#####################
    # print('f_name',f_name)

    for i,item in enumerate(npyfile):
        # print('i,item:',i,item[0],item[5])
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        save_path2 = os.path.join(save_path,f_name[i])
        save_path2 = save_path2.split('.')[0]+'.png'
        print(save_path2)
        cv2.imwrite(save_path2, img)
        # io.imsave(os.path.join(save_path,f_name[i]),img)

def clip_to_512(img_path,label_path,save_path,save_l_path):
    img = cv2.imread(img_path)
    label = cv2.imread(label_path)
    # print(label_path)
    if img.shape ==label.shape :
        cols = img.shape[0]
        rows = img.shape[1]
    else:
        print('clip to 512 时，标签和影像不对应！')
        sys.exit()
    cols_n = cols//512 + 1
    rows_n = rows//512 + 1
    top_liet_x = 0
    top_liet_y = -512
    for col_n in range(cols_n) :
        top_liet_x = 0
        top_liet_y += 512
        for row_n in range(rows_n):
            # print(top_liet_x+512<=cols,top_liet_y+512 <=rows)
            if top_liet_x+512>=cols :
                top_liet_x = cols -512
            if  top_liet_y+512 >=rows:
                top_liet_y = rows -512
            cropped = img[top_liet_x:top_liet_x+512,top_liet_y:top_liet_y+512]
            cropped_2 = label[top_liet_x:top_liet_x + 512, top_liet_y:top_liet_y + 512]
            cropped_2 = cv2.cvtColor(cropped_2, cv2.COLOR_BGR2GRAY)
            for i in range(512):
                for ii in range(512):
                    if cropped_2[i,ii] >= 20:
                        cropped_2[i,ii] = 255
                    else:
                        cropped_2[i,ii] = 0
            # a = os.path.basename(img_path).split('.')[0]
            b=os.path.join(save_path,os.path.basename(img_path).split('.')[0])+str(row_n)+str(col_n)+str('.png')
            b2=os.path.join(save_l_path,os.path.basename(label_path).split('.')[0])+str(row_n)+str(col_n)+str('.png')
            # print(b)
            cv2.imwrite(b,cropped)
            cv2.imwrite(b2, cropped_2)
            # print(b2)
            top_liet_x += 512
            # print("1",top_liet_x,top_liet_y)
            # elif top_liet_y+512 >=rows:
            #     top_liet_y = rows - 512
            #     cropped = img[top_liet_x:top_liet_x+512,top_liet_y:top_liet_y+512]
            #     b = os.path.join(save_path, os.path.basename(img_path).split('.')[0]) + str(row_n) + str(col_n) + str(
            #         '.png')
            #     cv2.imwrite(b,cropped)
            #     print('here')
            #     top_liet_x += 512
            # else:
            #     top_liet_x = rows - 512
            #     top_liet_y = cols - 512
            #     cropped = img[top_liet_x:top_liet_x+512,top_liet_y:top_liet_y+512]
            #     b = os.path.join(save_path, os.path.basename(img_path).split('.')[0]) + str(row_n) + str(col_n) + str(
            #         '.png')
            #     cv2.imwrite(b,cropped)
            #     print('3')

# def go_to_6b(path1,path2,savepath):
#     imgrgb = gdal.Open(path1)
#     im_width =imgrgb.RasterXSize #栅格矩阵的行数
#     im_height = imgrgb.RasterYSize #栅格矩阵的行数
#     band1 = imgrgb.GetRasterBand(1)
#     band2 = imgrgb.GetRasterBand(2)
#     band3 = imgrgb.GetRasterBand(3)
#     imgprg = gdal.Open(path2)
#     band4 = imgprg.GetRasterBand(1)
#     band5 = imgprg.GetRasterBand(2)
#     band6 = imgprg.GetRasterBand(3)
#     driver = gdal.GetDriverByName("GTiff")
#     dataset = driver.Create(savepath, im_width, im_height, 6, gdal.GDT_Float32)
#     dataset.GetRasterBand(1).WriteArray(band1)
#     dataset.GetRasterBand(2).WriteArray(band2)
#     dataset.GetRasterBand(3).WriteArray(band3)
#     dataset.GetRasterBand(4).WriteArray(band4)
#     dataset.GetRasterBand(5).WriteArray(band5)
#     dataset.GetRasterBand(6).WriteArray(band6)

def pca(bace_path):
    im_path = bace_path
    # print(im_path)
    img = cv2.imread(im_path, cv2.IMREAD_COLOR)
    # cv2.imshow("original", img)
    nrows, ncolumns = img.shape[0:2]  # 获取图片的行数与列数
    imgArrT = np.array(img, dtype=np.int64).reshape(-1, 3)  # 每行是一个像素点
    mean = np.mean(imgArrT, axis=0)  # 求所有像素点的平均值
    CenterImg = imgArrT - mean  # 将每个像素减去平均值，以中心化

    covar = np.matmul(np.transpose(CenterImg), CenterImg)
    evalues, evectors = np.linalg.eig(covar)  # 求特征值及特征向量
    principle_xis = evectors[:, np.argmax(evalues)]  # 特征值最大的轴是主轴
    principle_xis = principle_xis / np.sum(principle_xis)  # 将主轴的元素和变为1
    principle_xis = principle_xis[:, np.newaxis]  # 将主轴转换成（3,1）
    principle_img = np.matmul(imgArrT, principle_xis)  # 投影到主轴
    principle_img = np.uint8(principle_img.reshape((nrows, ncolumns)))  # 还原为主轴灰度图
    # cv2.imshow("principle", principle_img)

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", grayimg)
    # cv2.imwrite(im_save, principle_img)
    # print(im_save)
    # cv2.waitKey(0)
    return principle_img

