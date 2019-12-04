import os
import json
import time
import shutil

import random
from PIL import Image
import cv2
from albumentations import HorizontalFlip, VerticalFlip
import keras
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras import backend as K
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def Hflip(hf, image):
    image = hf(image=image)['image']
    return image


def Vflip(vf, image):
    image = vf(image=image)['image']
    return image


hf = HorizontalFlip(p=1)
vf = VerticalFlip(p=1)

# Import Training Data
train_df = pd.read_csv('train.csv')
# train_df = train_df.dropna()
# Import Training Images Names XXXXXX.jpg
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])

# print('Train/'+train_df[1:2]['ImageId'])

# Import Training Class Fish
train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

# Checks whether the label has encoded pixels by checking the cell contents
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

# Count the number of labels present in each image
mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

h = 320
w = 320
k = 0


def duck(k):
    print('Done! ==> ' + str(k))
    k += 1
    return k


def rle_to_mask_to_image(rle, img_width, img_height, test_img, fcuk, label):
    image = np.zeros((img_width * img_height, 1), dtype=np.uint8)
    for rle_ in rle:
        if (str(rle_) == 'nan'):
            continue
        pixels = [int(num) for num in str(rle_).split(' ')]
        pixels = np.asarray(pixels).reshape(-1, 2)
        for i in range(pixels.shape[0]):
            start = pixels[i][0] - 1
            final = pixels[i][0] + pixels[i][1]
            image[start:final, 0] = 255
    image = image.reshape(img_width, img_height)
    image = cv2.resize(image, (h, w))
    image = np.stack((image) * 3, -1)
    # cv2.imwrite('Masks/'+label+'_'+str(fcuk), image)
    if label == 'Flower':
        # cv2.imwrite('Flower/'+label+'_'+str(fcuk), image)
        cv2.imwrite('Flower_H/' + label + '_' + str(fcuk), Hflip(hf, image))
        cv2.imwrite('Flower_V/' + label + '_' + str(fcuk), Vflip(vf, image))
        cv2.imwrite('Flower_X/' + label + '_' + str(fcuk), Vflip(vf, Hflip(hf, image)))
    elif label == 'Gravel':
        # cv2.imwrite('Gravel/'+label+'_'+str(fcuk), image)
        cv2.imwrite('Gravel_H/' + label + '_' + str(fcuk), Hflip(hf, image))
        cv2.imwrite('Gravel_V/' + label + '_' + str(fcuk), Vflip(vf, image))
        cv2.imwrite('Gravel_X/' + label + '_' + str(fcuk), Vflip(vf, Hflip(hf, image)))
    elif label == 'Sugar':
        # cv2.imwrite('Sugar/'+label+'_'+str(fcuk), image)
        cv2.imwrite('Sugar_H/' + label + '_' + str(fcuk), Hflip(hf, image))
        cv2.imwrite('Sugar_V/' + label + '_' + str(fcuk), Vflip(vf, image))
        cv2.imwrite('Sugar_X/' + label + '_' + str(fcuk), Vflip(vf, Hflip(hf, image)))
    elif label == 'Fish':
        # cv2.imwrite('Fish/'+label+'_'+str(fcuk), image)
        cv2.imwrite('Fish_H/' + label + '_' + str(fcuk), Hflip(hf, image))
        cv2.imwrite('Fish_V/' + label + '_' + str(fcuk), Vflip(vf, image))
        cv2.imwrite('Fish_X/' + label + '_' + str(fcuk), Vflip(vf, Hflip(hf, image)))


def make_mask(val):
    if val:
        print("---In Mask---")
        for i in range(0, len(train_df.index), 1):
            fig = train_df[i:i + 1]['EncodedPixels']
            i_n = train_df[i:i + 1]['ImageId'].to_string(index=False).strip()
            label = train_df[i:i + 1]['ClassId'].to_string(index=False).strip()
            image = 'Train/' + train_df[i:i + 1]['ImageId'].to_string(index=False).strip()
            train_img = cv2.imread(image)
            rle_to_mask_to_image(fig, 2100, 1400, train_img, i_n, label)


X_Train = np.zeros((5546, h, w, 1))
Y_Train = np.zeros((5546, h, w, 1))

k = duck(k)


def conv(filter_, model):
    model = Conv2D(filter_, (3, 3), kernel_initializer="he_normal", padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('elu')(model)
    model = Conv2D(filter_, (3, 3), kernel_initializer="he_normal", padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('elu')(model)
    return model


def build_model(input_img):
    model_1 = conv(16, input_img)
    model_1x = MaxPooling2D((2, 2))(model_1)
    model_1x = Dropout(0.5)(model_1x)

    model_2 = conv(32, model_1x)
    model_2x = MaxPooling2D((2, 2))(model_2)
    model_2x = Dropout(0.5)(model_2x)

    model_3 = conv(64, model_2x)
    model_3x = MaxPooling2D((2, 2))(model_3)
    model_3x = Dropout(0.5)(model_3x)

    model_4 = conv(128, model_3x)
    model_4x = MaxPooling2D((2, 2))(model_4)
    model_4x = Dropout(0.5)(model_4x)

    model_5 = conv(256, model_4x)


    model_6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(model_5)
    model_6 = concatenate([model_6, model_4])
    model_6 = Dropout(0.5)(model_6)
    model_6 = conv(16, model_6)

    model_7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(model_6)
    model_7 = concatenate([model_7, model_3])
    model_7 = Dropout(0.5)(model_7)
    model_7 = conv(12, model_7)

    model_8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(model_7)
    model_8 = concatenate([model_8, model_2])
    model_8 = Dropout(0.5)(model_8)
    model_8 = conv(8, model_8)

    model_9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(model_8)
    model_9 = concatenate([model_9, model_1], axis=3)
    model_9 = Dropout(0.5)(model_9)
    model_9 = conv(4, model_9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(model_9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def train(resize):
    index = 0
    print("Training Begins")
    for i in range(0, len(train_df.index), 4):
        arr = train_df['Image_Label'][i:i + 1].to_string(index=False).split('_')
        label = arr[0].strip()
        inp = arr[0].strip()
        y_label = cv2.imread('Flower_X/Flower_' + label, 0).reshape(h, w,
                                                                    1)  # reshape to bring 1 in dimention...3 dimension
        if (resize):
            inp = cv2.resize(cv2.imread('Train/' + inp, 0), (w, h)).reshape(h, w, 1)
            cv2.imwrite('IM/' + label, inp)
            cv2.imwrite('IM_H/' + label, Hflip(hf, inp))
            cv2.imwrite('IM_V/' + label, Vflip(vf, inp))
            cv2.imwrite('IM_X/' + label, Vflip(vf, Hflip(hf, inp)))
        else:
            inp = cv2.imread('IM_X/' + inp, 0).reshape(h, w, 1)
        X_Train[index] = inp / 255
        Y_Train[index] = y_label / 255
        index += 1
    np.save('X_train_Flower_X', X_Train)
    print('X Saved')
    np.save('Y_train_Flower_X', Y_Train)
    print('Y Saved')
    X_train, X_valid, y_train, y_valid = train_test_split(X_Train, Y_Train, test_size=0.1)
    print('Data Splitted Successfully')
    print("Train Complete")
    return X_train, X_valid, y_train, y_valid


def load(xyz):
    print("Loading Arrays")
    X_train = np.load('X_train_Gravel.npy').astype('float32')
    Y_train = np.load('Y_train_Gravel.npy').astype('float32')
    X_Train, X_valid, y_Train, y_valid = train_test_split(X_train, Y_train, test_size=0.3)
    print('Data Splitted Successfully')
    print("Loading Complete")
    return X_Train, X_valid, y_Train, y_valid


make_mask(False)
X_train, X_valid, y_train, y_valid = load(False)
'''
for k in range(0, 10):
    imshow(X_train[k, ..., 0])
    plt.show()
    imshow(y_train[k, ..., 0])
    plt.show()
'''
k = duck(k)

input_img = Input((h, w, 1), name='img')
k = duck(k)
model = build_model(input_img)
k = duck(k)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
k = duck(k)
model.summary()

callbacks = [
    EarlyStopping(patience=2, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=0),
    ModelCheckpoint('model-Gravel.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

k = duck(k)
results = model.fit(X_train, y_train, batch_size=32, epochs=2, callbacks=callbacks, validation_data=(X_valid, y_valid))
# model.load_weights('model-Flower.h5')
# X_train = np.load('X_train_Flower_X.npy').astype('float32')
# Y_train = np.load('Y_train_Flower_X.npy').astype('float32')

# X_Train, X_valid, y_Train, y_valid = train_test_split(X_train, Y_train, test_size=0.3)
# results = model.fit(X_train, y_Train, batch_size=32, epochs=2, callbacks=callbacks,validation_data=(X_valid, y_valid))

# model.load_weights('model-tgs-salt.h5')

X_Test = np.zeros((3698, h, w, 1))
print("Begininning")
if(tr_comp):
inp1 = cv2.resize(cv2.imread('test_images/*.jpg', 0),(w, h)).reshape(h, w, 1)
    cv2.imwrite('IM_Test/',inp)
    X_Test[index] = inp1/255
np.save('X_test',X_Test)
print('X_test Saved')
print("Test Numpy completed")

for i in range(15):
    idx = random.randint(0, len(X_test))
    x = X_test[idx]
    x = np.expand_dims(x, axis=0)
    predict = model.predict(x, verbose=1)
    predict = (predict > 0.5).astype(np.uint8)
    imshow(np.squeeze(predict[0]))
    plt.show()
    imshow(X_test[idx, ..., 0])
    plt.show()