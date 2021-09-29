import tensorflow.compat.v1 as tf
from model2 import *
from data import *
from keras.callbacks import ModelCheckpoint
from data import clip_to_512
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
tf.enable_eager_execution()
tf.compat.v1.disable_eager_execution()

def totiff():
    bace_dir = r'E:\python\unet-master\data\membrane\train\image_original'
    bace_dir2 = r'E:\python\unet-master\data\membrane\train\label_original'
    f_names = os.listdir(bace_dir)
    for f_name in f_names:
        imagepath = os.path.join(bace_dir,f_name)
        f_name = f_name.split('.')[0] + str('.tif')
        lablepath = os.path.join(bace_dir2,f_name)

        clip_to_512(imagepath,lablepath,
                r'E:\python\unet-master\data\membrane\train\image',
                r'E:\python\unet-master\data\membrane\train\label')
print('tf 版本： ',tf.__version__)

def train_sarat():
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    myGene = trainGenerator(8,r'E:\python\unet-master\data\membrane_47\train_save\train50_80','image','label',data_gen_args,save_to_dir = None)
    # model = unet()
    model = att_r2_unet()
    # model = resnet()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=1000,epochs= 500,callbacks=[model_checkpoint])

    testGene = testGenerator("data/membrane_47/test50/image",num_image = 18)
    results = model.predict_generator(testGene,18,verbose=1)
    # print(results)
    saveResult("data/membrane_47/result",results)
#
# def resulet_only(img_number):
#     data_gen_args = dict(rotation_range=0.2,
#                         width_shift_range=0.05,
#                         height_shift_range=0.05,
#                         shear_range=0.05,
#                         zoom_range=0.05,
#                         horizontal_flip=True,
#                         fill_mode='nearest')
#     myGene = trainGenerator(3,r'E:\python\unet-master\data\membrane_mas\train','train_image','train_label',data_gen_args,save_to_dir = None)
#     model = unet()
#     model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#     # model.fit_generator(myGene,steps_per_epoch=150,epochs=2,callbacks=[model_checkpoint])
#
#     testGene = testGenerator('data/membrane_mas/train/test_image',num_image = img_number)
#     results = model.predict_generator(testGene, img_number, verbose=1)
#     saveResult("data/membrane_mas/result256", results)

train_sarat()
# resulet_only(10)#1036
# totiff()
