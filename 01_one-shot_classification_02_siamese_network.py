import os
import random
import numpy as np
import glob
import pickle as pkl
import shutil
# import h5py

# import cv2
# from PIL import Image

import time
from tqdm import tqdm
# from time import time

from math import ceil
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.transform import rotate, AffineTransform, warp, rescale
from skimage.util import random_noise

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Concatenate, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K

from statistics import mean


# ----------
# REFERENCE:
# Siamese Neural Networks for One-Shot Image Recognition
# http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
# https://github.com/asagar60/Siamese-Neural-Networks-for-One-shot-Image-Recognition


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# prepare data
#  - Training Set has 964 classes with 20 samples in each class.
#    Model needs 2 images to predict the output.
#    If we train our model on every possible combination of image pair, 
#    this will be over by 185,849,560 possible pairs.
#    We can reduce the number of image pairs.
#    Since every class has 20 samples,  say E, that makes E(E-1)/2 image pairs (same) per class. 
#    If there are C classes that makes C* E(E-1)/2 image pairs in total. 
#    These images represent same-class image pairs. 
#    For 964 classes , there will be 183,160 same image pairs.
# 
#  - We still need different-class image pairs. 
#    The siamese network should be given a 1:1 ratio of same-class and
#    different-class pairs to train on.
#    So, we will sample 183,160 image pairs (different-class) at random.
#    Paper suggested different sizes of training data ( 30K, 90K, 150K, 270K, 810K, 1350K ).
#
#  - functions:
#      - saveImagePaths
#      - generateTrainingPairs
#      - val_eval_split
#          - wA_test_pairs
#          - uA_test_pairs
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
# prepare data
# -----------------------------------------------------------------------------------------------------------

class prepare_data:

    def __init__(self, train_path, test_path, save_folder_proc):
    
        self.train_path = train_path
        self.test_path = test_path
        self.save_folder_proc = save_folder_proc

    # ----------------------------------------------------------------
    # Train_im_paths.pkl = image_paths[alphabet][character][img] (alphabet:30  character:depends  img:20)
    # totally 964 (alphabet * character) * 20 image = 19280 images
    # ----------------------------------------------------------------
    def saveImagePaths(self, setType='Train'):
        
        folder_path = self.train_path

        image_paths = []

        for i, _dir in enumerate(os.listdir(folder_path)):
            _dirpath = os.path.join(folder_path, _dir)
            dir_images = []
            for _subdir in os.listdir(_dirpath):
                _subdirpath = os.path.join(_dirpath, _subdir)
                img_path_list =  glob.glob(_subdirpath + '/*.png')
                dir_images.append(img_path_list)
            
            dir_images = np.array(dir_images)
            # dir_images = np.reshape(dir_images, (1,dir_images.shape[0], dir_images.shape[1]))
                
            image_paths.append(dir_images)
        # image_paths = np.asarray(image_paths)

        # # ----------
        # # 30 folders in train_path
        # print(len(image_paths))
        # # 1st path ('Futurama') have 26 characters
        # print(len(image_paths[0]))
        # # 'Futurama': 1st characters have 20 images
        # print(len(image_paths[0][0]))
        # # all, 964 characters  (totally 964 * 20 = 19280 images)
        # tmp_count = 0
        # for i in range(len(image_paths)):
        #     for _ in range(len(image_paths[i])):
        #         tmp_count += 1
        # print(image_paths[0][0][0])
        # # ----------

        self.file_name = f'{self.save_folder_proc}/{setType}_im_paths.pkl'
        
        with open(self.file_name, 'wb') as f:
            pkl.dump(image_paths, f)
            print(f'{self.file_name} : saved')
        
    # ----------
    def generateTrainingPairs(self, n = 183160):

        pairs, y = [],[]

        with open(self.file_name, 'rb') as f:
            image_paths = pkl.load(f)
        
        train_filename = f'{self.save_folder_proc}/training_file_{n}.pkl'
        
        total = 0
        # i:  30 Alphabets
        for i in tqdm(range(len(image_paths))):
            # j:  characters in each Alphabet
            for j in range(len(image_paths[i])):
                # k:  each image
                for k in range(len(image_paths[i][j])):

                    # if total == 2 * n = 366320 --> dump
                    if total == 2 * n:
                        X, y = np.array(pairs), np.array(y)
                        with open(train_filename, 'wb') as f:
                            pkl.dump((X,y), f)
                            print(f'{train_filename} : saved')
                        return
                        
                    path_1 = image_paths[i][j][k]
                    
                    for m in range(k + 1, len(image_paths[i][j])):

                        # y == 1:  same Alphabet AND Character with path_1
                        path_2 = image_paths[i][j][m]
                        pairs.append([path_1, path_2])
                        y.append(1)
                        
                        # y == 0:  Do Not Care Alphabet AND Character
                        path_2 = random.sample(list(random.sample(list(random.sample(list(image_paths),1)[0]),1)[0]),1)[0]
                        pairs.append([path_1, path_2])
                        y.append(0)
                        
                        total = total + 2


    # ------------------------------------------------------------------------------------------------
    # We need to have image pairs for " Within Alphabet and Unstructured Alphabet" for both validation and Evaluation Set.
    # We will split the default evaluation set into validation and Evaluation set.
    # The original Omniglot github repo uses within-alphabet evaluation. 
    # It defines 20 runs, each of which comprises 20 training images and 20 testing images from the same alphabet (2 runs for each of the 10 evaluation alphabets.
    # So, We split the data based on Alphabet sets used in these 20 runs. 
    # These are namely ['Atlantean', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada', 'Malayalam', 'Manipuri', 'Old_Church_Slavonic_(Cyrillic)' ,'Tengwar','Tibetan']
    # which we save as 10 split (10 Alphabet) evaluation set
    #
    # So we define :
    # wA_val_10_split_images.pkl : within alphabet pairs for validation set (10 Alphabets)
    # uA_val_10_split_images.pkl : Unstructured alphabet pairs for validation set (10 Alphabets)
    # wA_eval_10_split_images.pkl : within alphabet pairs for Evaluation set (10 Alphabets)
    # uA_eval_10_split_images.pkl : Unstructured alphabet pairs for Evaluation set (10 Alphabets)
    # wA_eval_20_split_images.pkl : within alphabet pairs for Evaluation set (20 Alphabets)
    # uA_eval_20_split_images.pkl : Unstructured alphabet pairs for Evaluation set (20 Alphabets)
    # ------------------------------------------------------------------------------------------------
    def val_eval_split(self):
    
        # THIS IS test_path
        folder_path = self.test_path

        # ----------    
        # eval: only 10 Alphabets
        eval_list = ['Atlantean', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada', 'Malayalam', 'Manipuri', 'Old_Church_Slavonic_(Cyrillic)' ,'Tengwar','Tibetan']

        # dir_list: all 20 Alphabets
        dir_list = os.listdir(folder_path)
        # print(len(dir_list))

        # val: dir_list - eval (remain 10 Alphabets)
        val_dir = [dir_ for dir_ in dir_list if dir_ not in eval_list]
        # ----------    
        
        #Type 1 - validation data , evaluation data
        self.wA_test_pairs(folder_path, val_dir, savefilename = f'{self.save_folder_proc}/wA_val_10_split_images.pkl', n_way = 20)
        self.uA_test_pairs(folder_path, val_dir, savefilename = f'{self.save_folder_proc}/uA_val_10_split_images.pkl',  n_way = 20)
        
        self.wA_test_pairs(folder_path, eval_list, savefilename = f'{self.save_folder_proc}/wA_eval_10_split_images.pkl',  n_way = 20)
        self.uA_test_pairs(folder_path, eval_list, savefilename = f'{self.save_folder_proc}/uA_eval_10_split_images.pkl',  n_way = 20)
        
        #Type 2 - Validation + evaluation
        self.wA_test_pairs(folder_path, eval_list, savefilename = f'{self.save_folder_proc}/wA_eval_20_split_images.pkl',  n_way = 20)
        self.uA_test_pairs(folder_path, eval_list, savefilename = f'{self.save_folder_proc}/uA_eval_20_split_images.pkl',  n_way = 20)
        

    # -------------------------------------------------------
    # Within-Alphabet:
    #   - Choose an alphabet, then choose K characters from that alphabet (without replacement).
    # -------------------------------------------------------
    def wA_test_pairs(self, folder_path, dirs, savefilename, n_way = 20):

        X,y = [],[]

        for alpha in dirs:
            alphabet_dir = os.path.join(folder_path, alpha)

            # choose n_way character
            char_dirs = os.listdir(alphabet_dir)
            char_dirs = random.sample(char_dirs, n_way)

            # ----------
            # 2 samples from each char dirs
            set_1, set_2 = [],[]
            for char in char_dirs:
                char_path = os.path.join(alphabet_dir, char)
                img_paths =  glob.glob(char_path + "/*.png")
                random_samples = random.sample(img_paths, 2)
                set_1.append(random_samples[0])
                set_2.append(random_samples[1])
                # ----------
                # assert len(set_1) == n_way
                # assert len(set_2) == n_way
                # ----------

            for i,imPath1 in enumerate(set_1):
                for j,imPath2 in enumerate(set_2):
                    img1 = np.expand_dims(mpimg.imread(imPath1), axis = 2)
                    img2 = np.expand_dims(mpimg.imread(imPath2), axis = 2)
                    X.append([img1, img2])

                    # 1:positive  0:negative
                    y.append(1 if i==j else 0)
            # ----------
            # 400 = 20 * 20
            # assert len(X) == 400
            # assert len(y) == 400
            # ----------
            
            for i,imPath1 in enumerate(set_2):
                for j,imPath2 in enumerate(set_1):
                    img1 = np.expand_dims(mpimg.imread(imPath1), axis = 2)
                    img2 = np.expand_dims(mpimg.imread(imPath2), axis = 2)
                    X.append([img1, img2])

                    # 1:positive  0:negative
                    y.append(1 if i==j else 0)
            # ----------
            # 400 + 400
            # assert len(X) == 800
            # assert len(y) == 800
            # ----------

        X, y = np.array(X), np.array(y)
        #y = np.reshape(y,(-1,1))

        if savefilename == None:
            return X,y
        else:         
            with open(savefilename, "wb") as f:
                pkl.dump((X, y), f)
                print(f'{savefilename} : saved')
	
    # -------------------------------------------------------
    # unstructured :
    #   - Concatenate the characters of all alphabets, then choose K characters (without replacement). 
    #     The hierarchy of alphabets and characters is ignored.
    # -------------------------------------------------------
    def uA_test_pairs(self, folder_path, dirs, savefilename, classes = None, n_way = 20):
            
        X, y = [],[]
        
        if classes == None:
            dirs = random.sample(dirs, len(dirs))
        else:
            dirs = random.sample(dirs, classes)
            
        for alpha in dirs:
            alphabet_dir = os.path.join(folder_path, alpha)
            char_dirs = os.listdir(alphabet_dir)
            char_dirs = random.sample(char_dirs,n_way)
            for char in char_dirs:
                
                char_path = os.path.join(alphabet_dir, char)
                img_paths =  glob.glob(char_path + "/*.png")
                
                imPath1, imPath2 = random.sample(img_paths,2)
                img1 = np.expand_dims(mpimg.imread(imPath1), axis = 2)
                img2 = np.expand_dims(mpimg.imread(imPath2), axis = 2)
                X.append([img1, img2])
                y.append(1)
                
                for _ in range(n_way-1):
                    random_alpha_pick = random.sample(dirs,1)[0]
                    random_alphabet_dir = os.path.join(folder_path, random_alpha_pick)
                    random_char_dirs = os.listdir(random_alphabet_dir)
                    random_pick = random.sample(random_char_dirs, 1)[0]
                                        
                    while(random_pick == char):
                        random_pick = random.sample(random_char_dirs,1)[0]
                    
                    random_char_dir = os.path.join(random_alphabet_dir, random_pick)
                    imPath2 = random.sample(glob.glob(random_char_dir + "/*.png"),1)[0]
                    img2 = np.expand_dims(mpimg.imread(imPath2), axis = 2)
                    X.append([img1, img2])
                    y.append(0)

        X, y = np.array(X), np.array(y)
        #y = np.reshape(y,(-1,1))
               
        if savefilename == None:
            return (X,y)
        else:
            with open(savefilename, "wb") as f:
                pkl.dump((X,y), f)
                print(f'{savefilename} : saved')


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# data loader
# -----------------------------------------------------------------------------------------------------------

# ----------
# transformation

def affinetransform(image):
    transform = AffineTransform(translation=(-30, 0))
    warp_image = warp(image,transform, mode="wrap")
    return warp_image

def anticlockwise_rotation(image):
    angle= random.randint(0, 45)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0, 45)
    return rotate(image, -angle)


def transform(image):
    if random.random() > 0.5:
        image = affinetransform(image)
    if random.random() > 0.5:
        image = anticlockwise_rotation(image)
    if random.random() > 0.5:
        image = clockwise_rotation(image)

    return image


# ----------
def pkl_data(filename):
    with open(filename,'rb') as f:
        X_t, y_t = pkl.load(f)
    return X_t, y_t


class data_gen:

    def __init__(self, batch_size = 32, isAug = True):
        self.batch_size = batch_size
        self.isAug = isAug

    def load_data_batch(self, training_file):

        X,y = pkl_data(training_file)
        load_batch = 1024
        train_len = len(X)

        while(True):
            for i in range(int(train_len / load_batch)):
                start = i * load_batch
                end = (i + 1) * load_batch if i != int(train_len / load_batch) else -1
                X_t = X[start:end]
                y_t = y[start:end]
                X_t, y_t = shuffle(X_t, y_t, random_state=0)

                for offset in range(0, load_batch, self.batch_size):
                    X_left, X_right, _y = X_t[offset:offset + self.batch_size, 0], X_t[offset:offset + self.batch_size, 1], y_t[offset:offset + self.batch_size]

                    #X_left, X_right, y = X_t[offset:offset + 5,0], X_t[offset:offset + 5,1], y_t[offset:offset + 5]

                    X_left_batch = []
                    X_right_batch = []
                    y_batch = []

                    for i in range(len(X_left)):
                        if random.random() > 1024:
                            # ----------------------
                            # apply transform
                            # ----------------------
                            X_i = np.expand_dims(transform(mpimg.imread(X_left[i])), axis = 2)
                            X_j = np.expand_dims(transform(mpimg.imread(X_right[i])), axis = 2)

                            X_left_batch.append(X_i)
                            X_right_batch.append(X_j)
                            y_batch.append(_y[i])
                        else:
                            # ----------------------
                            # no apply transform
                            # ----------------------
                            X_i = np.expand_dims(mpimg.imread(X_left[i]), axis = 2)
                            X_j = np.expand_dims(mpimg.imread(X_right[i]), axis = 2)

                            X_left_batch.append(X_i)
                            X_right_batch.append(X_j)
                            y_batch.append(_y[i])

                    X_left_batch, X_right_batch, y_batch = np.asarray(X_left_batch), np.asarray(X_right_batch), np.asarray(y_batch)
                    X_left_batch, X_right_batch, y_batch  = shuffle(X_left_batch, X_right_batch, y_batch, random_state = 0)

                    #print("print_shape",X_left_batch.shape, X_right_batch.shape, y_batch.shape)
                    #print(X_left_batch[0], X_right_batch[1])

                    yield [X_left_batch, X_right_batch], y_batch


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# validation accuracy
# -----------------------------------------------------------------------------------------------------------

def test_pairs(model_obj, file_name, n_way = 20):

    correct_pred = 0

    X, y = pkl_data(file_name)
    #print(X.shape, y.shape)

    # j = 0

    for i in range(0, len(X), n_way):
        X_left, X_right, _y  = X[i: i+n_way, 0], X[i: i+n_way, 1], y[i : i+n_way]
        #X_left, X_right, y = sub_data_X[:,0], sub_data_X[:,1], sub_data_y
        X_left, X_right, _y = np.array(X_left), np.array(X_right), np.array(_y)

        # test one shot
        prob = model_obj.predict([X_left, X_right])

        #### here NOT y but _y !!! ####
        if np.argmax(prob) == np.argmax(_y):
            correct_pred += 1
        ###############################

    acc =  correct_pred * 100 / (len(X) / n_way)
    return acc


def test_validation_acc(model_obj, wA_file, uA_file, n_way=20):
    wA_acc = test_pairs(model_obj, wA_file, n_way)
    uA_acc = test_pairs(model_obj, uA_file, n_way)
    return (wA_acc, uA_acc)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# def continue_training(self, save_folder):

#     with open(f'{save_folder}/best_model/model_details.pkl', 'rb') as f:
#         model_details = pkl.load(f)

#     with open(self.val_acc_filename, "rb") as f:
#         self.v_acc,self.train_metrics  = pkl.load(f)

#     self.best_acc = model_details['acc']
#     self.start = model_details['iter'] + 1
#     K.set_value(self.model.optimizer.learning_rate, model_details['model_lr'])
#     K.set_value(self.model.optimizer.momentum, model_details['model_mm'])
#     best_model = f'{save_folder}/best_model/best_model.h5'
#     self.model.load_weights(best_model)
#     print('\n\n----------------------------------------------------Loading saved Model----------------------------------------------------\n\n')


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# base setting
# -----------------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/omniglot/one-shot'

data_folder = '/media/kswada/MyFiles/dataset/omniglot'
# data_folder = '/home/kswada/kw/omniglot/one-shot/01_data/omniglot'


# -----------------------------------------------------------------------------------------------------------
# data preprocess / preparation
# -----------------------------------------------------------------------------------------------------------

train_path = os.path.join(data_folder, 'images_background')

test_path = os.path.join(data_folder, 'images_evaluation')

save_folder_proc = './01_data/siamese_network'


# ----------
prep_data = prepare_data(
    train_path = train_path,
    test_path = test_path,
    save_folder_proc = save_folder_proc
)


prep_data.saveImagePaths()

prep_data.generateTrainingPairs()

prep_data.val_eval_split()


# -----------------------------------------------------------------------------------------------------------
# check
# -----------------------------------------------------------------------------------------------------------

# ----------
# 1. image paths
filename = f'{save_folder_proc}/Train_im_paths.pkl'

with open(filename,'rb') as f:
    data = pkl.load(f)

print(f'file : {filename}')
print(f'  len(data)     :{len(data)}')
print(f'  data[0][0][0] :{data[0][0][0]}')


# ----------
# 2. data for train
train_filename = f'{save_folder_proc}/training_file_183160.pkl'

with open(train_filename,'rb') as f:
    data = pkl.load(f)

print(f'file : {train_filename}')
print(f'  len(data)     :{len(data)}')

# 366320 (= 183160 * 2)
print(f'  len(data[0])  :{len(data[0])}')
print(f'  len(data[1])  :{len(data[1])}')

for i in range(2):
    j = i * 2
    # 1:  Same Alphabet and Same Character pair but different image
    print(f'{i} - {j}  :  {data[0][j]} : {data[1][j]}')
    # 0:  Same Alphabet but different Character
    print(f'{i} - {j+1}:  {data[0][j+1]} : {data[1][j+1]}')


# ----------
# 3. images

wA_val_10 = f'{save_folder_proc}/wA_val_10_split_images.pkl'
uA_val_10 = f'{save_folder_proc}/uA_val_10_split_images.pkl'
wA_eval_10 = f'{save_folder_proc}/wA_eval_10_split_images.pkl'
uA_eval_10 = f'{save_folder_proc}/uA_eval_10_split_images.pkl'
wA_eval_20 = f'{save_folder_proc}/wA_eval_20_split_images.pkl'
uA_eval_20 = f'{save_folder_proc}/uA_eval_20_split_images.pkl'

file_list = [wA_val_10, uA_val_10, wA_eval_10, uA_eval_10, wA_eval_20, uA_eval_20]

for f in file_list:
    data = pkl_data(f)
    print(f'file : {f}')
    print(f'  data[0].shape :{data[0].shape}')

    # only 5% of pairs is from same character
    print(f'  data[1].shape :{data[1].shape}   sum(data[1]) :{sum(data[1])}')

# -->
# Within-Alphabet (wA):  (8000, 2, 105, 105, 1), (8000,)
# Unstructured-Alphabet (uA):  (4000, 2, 105, 105, 1), (4000,)

print(data[0][0][0].shape)
print(data[1])

# ----------
idx = np.where(data[1]==1)[0]
print(idx)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# siamese network model
# -----------------------------------------------------------------------------------------------------------

def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x-y), axis = 1, keepdims = True)
    result = K.maximum(sum_square, K.epsilon())
    return result


# ----------
# Paper suggested using L1 distance while designing the model,
# but I got slightly better result on L2 distance.
# On L1 accuracy was around 72.75% on Within alphabet and 86 % on unstructured alphabet pairs.
# ----------

def l1_dist(vect):
    x, y = vect
    return K.abs(x-y)


# ----------
def siamese_network(initial_learning_rate=0.001, batch_size=32):

    W_init_1 = RandomNormal(mean=0, stddev=0.01)
    b_init = RandomNormal(mean=0.5, stddev = 0.01)
    W_init_2 = RandomNormal(mean=0, stddev=0.2)

    input_shape = (105, 105, 1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()
    convnet.add(Conv2D(64, (10,10),activation='relu',input_shape=input_shape, kernel_initializer=W_init_1, bias_initializer = b_init ,kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (7,7),activation='relu', kernel_initializer=W_init_1, bias_initializer = b_init ,kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (4,4),activation='relu', kernel_initializer=W_init_1, bias_initializer = b_init ,kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256, (4,4),activation='relu', kernel_initializer=W_init_1, bias_initializer = b_init ,kernel_regularizer=l2(2e-4)))
    convnet.add(Flatten())
    convnet.add(Dense(4096, activation="sigmoid", kernel_initializer=W_init_2, bias_initializer = b_init ,kernel_regularizer=l2(1e-3)))
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)


    # distance: here euclidena distance or L1 distance
    merge_layer = Lambda(euclidean_dist)([encoded_l, encoded_r])
    # merge_layer = Lambda(l1_dist)([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid')(merge_layer)

    model = Model(inputs=[left_input, right_input], outputs=prediction)


    # ----------
    # if momentum is large (such as 0.9), does not work well ...
    optimizer = SGD(learning_rate = initial_learning_rate, momentum = 0.5)

    # initial_learning_rate = 0.001
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #                                                             initial_learning_rate=initial_learning_rate,
    #                                                             decay_steps=10000,
    #                                                             decay_rate=0.96,
    #                                                             staircase=True)

    # optimizer = SGD(learning_rate = lr_schedule)

    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    return model


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# train
# 
#   - There are two different ways to obtain few-shot classification problems for testing an algorithm. 
#     We will refer to these as "within-alphabet" (wA) and "unstructured" (uA) evaluation.
#     The difference lies in how a random set of K classes is obtained:
#
#   - To have better accuracy of the model, 
#     we will test the model on "Within Alphabet" and "unstructured Alphabet" Pairs.
#     This strategy is inspired by Jack Valmadre's few-shot classification
#     repo - https://github.com/jvlmdr/omniglot-eval
#
# within-alphabet (wA) : Choose an alphabet, then choose K characters from that alphabet (without replacement).
# unstructured (uA): Concatenate the characters of all alphabets, then choose K characters (without replacement).
# The hierarchy of alphabets and characters is ignored.
# Intuitively, we might expect that the unstructured problem is easier, 
# because there is likely to be more variation between alphabets than within alphabets.
# (This may seem counter-intuitive since characters within an alphabet must be different
# from one another, whereas characters across alphabets may be identical.
# However, a character in one alphabet can have at most one such near-identical match in another alphabet.)
# -----------------------------------------------------------------------------------------------------------

seed = 0

np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


save_folder = './04_output/siamese_network'
save_folder_bestmodel = './04_output/siamese_network/best_model'
wA_file = f'{save_folder_proc}/wA_val_10_split_images.pkl'
uA_file = f'{save_folder_proc}/uA_val_10_split_images.pkl'


# ----------
batch_size = 32
initial_learning_rate = 0.001

model = siamese_network(
    initial_learning_rate=initial_learning_rate,
    batch_size=batch_size)


# ----------
# save model json
model_json = model.to_json()

with open(f'{save_folder}/model.json', 'w') as json_file:
    json_file.write(model_json)

# ----------
# model details
model_details = {}
model_details['acc'] = 0
model_details['iter'] = 0
model_details['model_lr'] = 0.0
model_details['model_mm'] = 0.0


# ----------
# data generator
data_generator = data_gen(batch_size, isAug=True)

training_file = f'{save_folder_proc}/training_file_183160.pkl'

train_generator = data_generator.load_data_batch(training_file=training_file)

X_batch, y_batch = next(train_generator)

print(len(X_batch))
print(X_batch[0].shape)
print(X_batch[1].shape)
print(len(y_batch))
print(y_batch)


# ----------
start = 1
#182000
#216000
n_iteration = 1000000
# n_iteration = 5000
save_interval = 500

v_acc = []
train_metrics = []
best_acc = 0
train_loss, train_acc = [],[]

linear_inc = 0.01


# ----------
shutil.rmtree(f'{save_folder_bestmodel}')
os.makedirs(f'{save_folder_bestmodel}')

for i in range(start, n_iteration):

    start_time = time.time()

    X_batch, y_batch = next(train_generator)
    #print(X_batch[0].shape,X_batch[1].shape, y_batch.shape)
    #print(type(X_batch), type(y_batch))
    #return

    # ----------
    # train on batch
    loss = model.train_on_batch(X_batch, y_batch)
    train_loss.append(loss[0])
    train_acc.append(loss[1])

    # ----------
    if i % save_interval == 0:
        train_loss = mean(train_loss)
        train_acc = mean(train_acc)
        train_metrics.append([train_loss, train_acc])

        #loss_data.append(loss)

        # ----------
        # validation accuracy
        val_acc  = test_validation_acc(model_obj=model, wA_file=wA_file, uA_file=uA_file, n_way=20)
        #val_acc = [wA_acc, uA_acc]
        v_acc.append(val_acc)

        # ----------
        # save if val_acc[0] (= within-alphabet accuracy) > best_acc
        if val_acc[0] > best_acc:

            print('\n***Saving model***\n')
            #self.model.save_weights(f'{save_folder_bestmodel}/model_{i}_val_acc_{val_acc[0]}.h5')
            model.save_weights(f'{save_folder_bestmodel}/best_model_{i}_{val_acc[0]}.h5')
            model_details['acc'] = val_acc[0]
            model_details['iter'] = i
            model_details['model_lr'] = K.get_value(model.optimizer.learning_rate)
            model_details['model_mm'] = K.get_value(model.optimizer.momentum)            

            # ----------
            best_acc = val_acc[0]

            # ----------
            # save validation accuracy and model details
            with open(f'{save_folder_bestmodel}/val_acc', 'wb') as f:
                pkl.dump((v_acc, train_metrics), f)

            with open(f'{save_folder_bestmodel}/model_details.pkl', 'wb') as f:
                pkl.dump(model_details, f)

        end_time = time.time()
        print(f'Iteration :{i}')
        print(f'   - learning rate: {K.get_value(model.optimizer.learning_rate):.8f}')
        print(f'   - momentum     : {K.get_value(model.optimizer.momentum):.6f}')
        print(f'   - avg_loss     : {train_loss:.4f}   avg_acc   : {train_acc:.4f}')
        print(f'   - wA_acc       : {val_acc[0]:.2f}   u_Acc     : {val_acc[1]:.2f}')
        print(f'   - taken time   : {end_time - start_time:.2f} s')

        train_loss, train_acc = [], []

    # --------------------------------------
    # update learning rate and momentum
    # Learning rate were decayed by 1 percent per 5000 iterations.
    # ( Paper suggested decaying learning rate per epoch, 
    # where epoch here corresponds to 1 iteration over the training set,
    # which is almost same as our approach). 
    # We linearly increase the momentum by 0.01 per 5000 iterations till it converges to 0.9.
    # --------------------------------------
    if i % 5000 == 0:
        K.set_value(model.optimizer.learning_rate, K.get_value(model.optimizer.learning_rate) * 0.99)
        # K.set_value(model.optimizer.momentum, min(0.9, K.get_value(model.optimizer.momentum) + linear_inc))


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# plot validation accuracy
# -----------------------------------------------------------------------------------------------------------

from scipy.interpolate import make_interp_spline, BSpline

def plot_metric(loss, acc):
    
    size = len(acc)
    x = np.array(range(1,size+1))
    xnew = np.linspace(1,size,100)

    spl = make_interp_spline(x, acc, k=3) #BSpline object
    ynew = spl(xnew)
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.plot(acc,color = 'b', alpha=0.4, label = 'Avg Batch Accuracy')
    plt.plot(xnew, ynew,color = 'b', alpha=1, label = 'Smooth Average Accuracy')
    plt.xlabel('Steps (for every 500)')
    plt.ylabel('Accuracy')
    plt.legend(loc = "upper left")
    plt.title('Accuracy')

    spl = make_interp_spline(x, loss, k=3) #BSpline object
    ynew = spl(xnew)
    plt.subplot(1,2,2)
    plt.plot(loss,color = 'b', label = 'Avg Batch Loss')
    #plt.plot((xnew, ynew),color = 'b', alpha=1)
    plt.xlabel('Steps (for every 500)')
    plt.ylabel('Loss')
    plt.legend(loc = "upper left")
    plt.title('Loss')
    plt.show()


# ----------
with open(f'{save_folder_bestmodel}/val_acc','rb') as f:
    v_acc, train_metrics = pkl.load(f)

train_metrics = np.array(train_metrics)

plot_metric(train_metrics[:,0], train_metrics[:,1])


# -----------------------------------------------------------------------------------------------------------
# Plotting Validation Accuracy on 20 - way one shot
# -----------------------------------------------------------------------------------------------------------

v_acc = np.array(v_acc)

x = np.array(range(1,len(v_acc) + 1))
xnew = np.linspace(1,len(v_acc), 100)

spl = make_interp_spline(x, v_acc[:,0], k=3) #BSpline object
ynew1 = spl(xnew)
spl = make_interp_spline(x, v_acc[:,1], k=3) #BSpline object
ynew2 = spl(xnew)
plt.figure(figsize=(10,10))
plt.plot(v_acc[:,0], color = 'b', alpha=0.4,  label = 'within alphabet pairs')
plt.plot(xnew, ynew1, color = 'b', alpha=1, label = 'smoothed within alphabet pairs')
plt.plot(v_acc[:,1], color = 'r', alpha=0.4,  label = 'unstructed alphabets pairs')
plt.plot(xnew, ynew2, color = 'r', alpha=1, label = 'smoothed unstructed alphabets pairs')
plt.xlabel('Steps (for every 500)')
plt.ylabel('Accuracy')
plt.legend(loc = "best")
plt.title('20-way Accuracy')

plt.show()


# ----------
# Training Loss and Accuracy on Training Set
print(train_metrics[-1])


# 20-way Within-Alphabet Pairs Accuracy and Unstructured Alphabet Pairs Accuracy
print(v_acc[-1])


# -----------------------------------------------------------------------------------------------------------
# Testing N-Way one shot
# -----------------------------------------------------------------------------------------------------------

# load model
batch_size = 32
initial_learning_rate = 0.001

model = siamese_network(
    initial_learning_rate=initial_learning_rate,
    batch_size=batch_size)


# ----------
best_model = 'best_model_163500_58.0.h5'

fpath_best_model = f'{save_folder_bestmodel}/{best_model}'

model.load_weights(fpath_best_model)


# ----------
# evaluation on 10 Alphabets (evaluation set)

wA_file = f'{save_folder_proc}/wA_eval_10_split_images.pkl'
uA_file = f'{save_folder_proc}/uA_eval_10_split_images.pkl'

wA_acc, uA_acc = test_validation_acc(model_obj=model, wA_file=wA_file, uA_file=uA_file, n_way=20)

# Within-Alphabet Pairs Accuracy for 20-way one shot sample:
print(wA_acc)

# Unstructured-Alphabet Pairs Accuracy for 20-way one shot sample:
print(uA_acc)


# ----------
# evaluation on 20 Alphabets (image evaluation folder)

wA_file = f'{save_folder_proc}/wA_eval_20_split_images.pkl'
uA_file = f'{save_folder_proc}/uA_eval_20_split_images.pkl'

wA_acc, uA_acc = test_validation_acc(model_obj=model, wA_file=wA_file, uA_file=uA_file, n_way=20)

# Within-Alphabet Pairs Accuracy for 20-way one shot sample:
print(wA_acc)

# Unstructured-Alphabet Pairs Accuracy for 20-way one shot sample:
print(uA_acc)


# -----------------------------------------------------------------------------------------------------------
# N-way One-Shot Testing
#   - We will now use a base model just to compare our model against this model.
#     This is inspired by Soren Bouma's implementation of Nearest Neighbour pairs.
#   - Our Aim is show that results from siamese net are far better than our base model.
# -----------------------------------------------------------------------------------------------------------

folder_path = test_path


# ----------
dir_list = os.listdir(folder_path)
print(dir_list)

dl = prepare_data()


# ----------
def test_one_shot(model_obj, X_left, X_right, y):
    prob = model_obj.predict([X_left, X_right])
    """
    print(prob)
    print(np.argmax(prob))
    print(np.argmax(y))
    return
    """
    if np.argmax(prob) == np.argmax(y):
        return 1
    else:
        return 0


def nearest_neighbour_correct(pairs, targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    X_left, X_right = pairs
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sqrt(abs(np.sum(X_left[i]**2 - X_right[i]**2)))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def test_data_pairs(n_way=20, wA = True):
    if wA:
        pairs = dl.wA_test_pairs(folder_path=folder_path, dirs=dir_list, savefilename=None, n_way=n_way)
    else:
        pairs = dl.uA_test_pairs(folder_path=folder_path, dirs=dir_list, savefilename=None, classes=None, n_way=n_way)
    
    X, y = pairs
    correct_pred = 0
    nn_correct = 0
    j = 0
    for i in range(0,len(X), n_way):
        X_left, X_right, _y = X[i: i+n_way, 0], X[i: i+n_way, 1], y[i : i+n_way]
        X_left, X_right, _y = np.array(X_left), np.array(X_right), np.array(_y)

        correct_pred += test_one_shot(X_left, X_right, _y)
        nn_correct += nearest_neighbour_correct((X_left, X_right), _y)

    acc =  correct_pred * 100 / (len(X) / n_way)
    nn_acc = nn_correct * 100 / (len(X) / n_way)
    return acc, nn_acc


def one_shot_accuracy():
    within_accuracies = []
    unstructred_accuracies = []
    #[2, 5, 6, 10, 15, 16, 20]
    for i in tqdm(range(2, 21)):
        within_accuracies.append(test_data_pairs(n_way=i, wA=True))
        unstructred_accuracies.append(test_data_pairs(n_way=i, wA=False)) 
    return within_accuracies ,unstructred_accuracies


# ----------
within_accuracies ,unstructred_accuracies = one_shot_accuracy()
print(within_accuracies)
print(unstructred_accuracies)

# ----------
ways = np.arange(2, 21, 1)

plt.figure(figsize = (10,10))
plt.plot(ways,np.asarray(within_accuracies)[:,0], color = 'b',  label = 'wA 20-way one shot accuracy')
plt.plot(ways,np.asarray(unstructred_accuracies)[:,0], color = 'r', label = 'uA 20-way one shot accuracy')
plt.plot(ways,np.asarray(within_accuracies)[:,1], color = 'g',  label = 'wA 20-way nearest neighbour accuracy')
plt.plot(ways,np.asarray(unstructred_accuracies)[:,1], color = 'm', label = 'uA 20-way nearest neighbour accuracy')
plt.xlabel('Steps (for every 500)')
plt.xticks(np.arange(2, 22, step=2))
plt.ylabel('N-way Accuracy')
plt.legend(loc = "best")
plt.title('N-way Accuracies')
plt.show()


# -----------------------------------------------------------------------------------------------------------
# Visualizing N-way pairs
# -----------------------------------------------------------------------------------------------------------

# load model
batch_size = 32
initial_learning_rate = 0.001

model = siamese_network(
    initial_learning_rate=initial_learning_rate,
    batch_size=batch_size)


# ----------
best_model = 'best_model_163500_58.0.h5'

fpath_best_model = f'{save_folder_bestmodel}/{best_model}'

model.load_weights(fpath_best_model)


# ----------
from mpl_toolkits.axes_grid1 import ImageGrid
import re

def generate_img_matrix(model_obj, X_left, X_right, y):
    X_left, X_right, _y = np.array(X_left), np.array(X_right), np.array(y)
    pred = model_obj.predict([X_left, X_right])
    index = np.argmax(pred)
    
    img0 = np.squeeze(X_left[0], axis = 2)
    X_p = []
    img_matrix = []
    
    for i in range(len(X_right)):
        img1 = np.squeeze(X_right[i], axis = 2)
        X_p.append(img1)
        if len(X_p) == 5:
            X_p =np.vstack(X_p)
            img_matrix.append(X_p)
            X_p = []
            
    img_matrix = np.asarray(img_matrix)
    img_matrix = np.hstack(img_matrix)
    return img0, img_matrix, index


def visualize_n_way(model_obj, file, n_way=20):

    with open(file, 'rb') as f:
        X,y = pkl.load(f)

    i = random.randint(0,int(len(X)/n_way))

    X_left, X_right,_y  = X[i: i+n_way,0],X[i: i+n_way,1], y[i : i+n_way]
    img0, img_matrix, index= generate_img_matrix(model_obj, X_left, X_right, _y)

    f, ax =  plt.subplots(1,3, figsize = (20,20))
    f.tight_layout()
    ax[0].imshow(img0, cmap = 'gray')
    ax[0].set_title('Test Image')
    ax[1].imshow(img_matrix, cmap = 'gray')
    ax[1].set_title('Support Set')
    ax[2].imshow(np.squeeze(X_right[index], axis = 2), cmap = 'gray')
    ax[2].set_title('Image with highest similarity in Support Set')
    plt.show()


# -----------
wA_file = f'{save_folder_proc}/wA_eval_20_split_images.pkl'
uA_file = f'{save_folder_proc}/uA_eval_20_split_images.pkl'

for _ in range(5):
    visualize_n_way(model_obj=model, file=wA_file, n_way=20)

for _ in range(5):
    visualize_n_way(model_obj=model, file=uA_file, n_way=20)
