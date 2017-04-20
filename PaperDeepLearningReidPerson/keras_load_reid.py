import matplotlib.pyplot as plt
from random import randint
import os
import numpy as np
np.random.seed(1217)
import h5py
from PIL import Image
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge,Dropout,Merge
from keras.regularizers import l2
from keras.optimizers import RMSprop, SGD, Adam, adadelta
from keras.preprocessing import image as pre_image
from PIL import Image as pil_image

K.set_image_dim_ordering('th')
optimizer = RMSprop(lr=1e-4)

def get_files(type_dir):
    files = [os.path.join(type_dir, file_i)
             for file_i in os.listdir(type_dir)
                if '.png' in file_i]
    return files

def get_imgs(type_dir):
    return [pil_image.open(f_i) for f_i in get_files(type_dir)]

def img_to_array(img):
    x = np.asarray(img, dtype='float32');
    x = x.reshape((3,x.shape[0], x.shape[1]))
    return x

def process_images(images):
    tam_images = len(images);

    proc_images1 = [];
    proc_images2 = [];
    proc_label = [];

    for i in range(tam_images):
        j = i + 1;
        while j < tam_images:
            aux_label = [];
            if i // 4 == j // 4:
                proc_label.append(1);
            else:
                break;
            #proc_label.append(aux_label);
            proc_images1.append(images[i]);
            proc_images2.append(images[j]);
            j = j + 1;
        if j + 4 < tam_images:
            for k in range(4):
                proc_label.append(0);
                proc_images1.append(images[i]);
                proc_images2.append(images[randint(j, tam_images - 1)]);

    print(proc_images1[0].shape);
    return proc_label,np.array(proc_images1),np.array(proc_images2);


def concat_iterat(input_tensor):
    input_expand = K.expand_dims(K.expand_dims(input_tensor, -1), -1)
    x_axis = []
    y_axis = []
    for x_i in range(5):
        for y_i in range(5):
            y_axis.append(input_expand)
        x_axis.append(K.concatenate(y_axis, axis=3))
        y_axis = []
    return K.concatenate(x_axis, axis=2);


def cross_input_asym(X):
    tensor_left = X[0]
    tensor_right = X[1]
    x_length = K.int_shape(tensor_left)[2]
    y_length = K.int_shape(tensor_left)[3]

    cross_y = []
    cross_x = []
    tensor_left_padding = K.spatial_2d_padding(tensor_left, padding=(2, 2),dim_ordering='th')
    tensor_right_padding = K.spatial_2d_padding(tensor_right, padding=(2, 2),dim_ordering='th')

    #print('tensor_left_padding:', K.int_shape(tensor_left)[0], K.int_shape(tensor_left)[1],K.int_shape(tensor_left)[2],K.int_shape(tensor_left)[3])

    for i_x in range(2, x_length + 2):
        for i_y in range(2, y_length + 2):
            cross_y.append(tensor_left_padding[:,:, i_x - 2:i_x + 3, i_y - 2:i_y + 3]-
                           - concat_iterat(tensor_right_padding[:,:, i_x, i_y]))
        cross_x.append(K.concatenate(cross_y,axis=3))
        cross_y = []
    cross_out = K.concatenate(cross_x,axis=2)
    #print('Dimensiones:',K.int_shape(cross_out)[0],' ',K.int_shape(cross_out)[1],' ',K.int_shape(cross_out)[2],' ',K.int_shape(cross_out)[3]);

    return K.abs(cross_out)


def cross_input_shape(input_shapes):
    input_shape = input_shapes[0]
    print('cross_input_shape:' ,input_shape[0],' ',input_shape[1],' ',input_shape[2] * 5,' ',input_shape[3]*5);
    return (input_shape[0], input_shape[1], input_shape[2]*5, input_shape[3]*5)


ximg_train=get_imgs('train');
ximg_test=get_imgs('test');

tamtrain=1000;
tamtest=100;

shapetrain = [tamtrain, 3,160, 60,];
shapetest = [tamtest,3,160,60];
img_train= np.zeros( (shapetrain) );
img_test= np.zeros( (shapetest));


for i in range(tamtrain):
    img_train[i]=img_to_array(ximg_train[i]);

for i in range(tamtest):
    img_test[i]=img_to_array(ximg_test[i]);

img_train=img_train.astype('float32')/255;
img_test=img_test.astype('float32')/255;

print (img_train.shape);
print (img_test.shape);

print("Size of:")
print("- Training-set:\t\t{}".format(tamtrain));
print("- Test-set:\t\t{}".format(tamtest));

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 20         # There are 20 of these filters.
# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 25         # There are 36 of these filters.

batch_size=100;
img_size1 = 60
img_size2 = 160;
img_size_flat = img_size1*img_size2;
img_shape = (img_size1, img_size2)
num_channels = 3
num_classes = 2
print ('ini model');
weight_decay=0.005;
a1 = Input(shape=(3,160, 60))
b1 = Input(shape=(3,160, 60))
a2 = Convolution2D(20, 5, 5, border_mode='same', activation='relu')(a1)
b2 = Convolution2D(20, 5, 5, border_mode='same', activation='relu')(b1)
a4 = MaxPooling2D(pool_size=(2, 2))(a2)
b4 = MaxPooling2D(pool_size=(2, 2))(b2)
a5 = Convolution2D(25, 5, 5, border_mode='same', activation='relu')(a4)
b5 = Convolution2D(25, 5, 5, border_mode='same', activation='relu')(b4)
a7 = MaxPooling2D(pool_size=(2, 2))(a5)
b7 = MaxPooling2D(pool_size=(2, 2))(b5)
a8 = merge([a7, b7], mode=cross_input_asym, output_shape=cross_input_shape)
a9 = Convolution2D(25, 5, 5, border_mode='same', activation='relu')(a8)
a11 = MaxPooling2D(pool_size=(2, 2))(a9)
a12 = Convolution2D(25, 3, 3, border_mode='same' , activation='relu' )(a11)
a13 = MaxPooling2D(pool_size=(2, 2))(a12)
c2 = Flatten()(a13)
c3 = Dense(256, activation='relu')(c2)
c33=Dropout(0.5)(c3)
c4 = Dense(1, activation='sigmoid')(c33)
print ('fin model');

model = Model(input=[a1, b1], output=c4)
model.compile(loss='binary_crossentropy',optimizer=SGD(lr=1e-4, momentum=0.9),metrics=['accuracy']);

xtrain_label,train_images1,train_images2 = process_images(img_train);
xtest_label,test_images1,test_images2 = process_images(img_test);

train_label=xtrain_label;
test_label=xtest_label;

model.load_weights('modelkeras_final_w10.h5');
tam_images = len(img_test);
print ('total imagenes: ',tam_images);

plt.imshow(ximg_test[0]);
plt.show();
plt.imshow(ximg_test[2]);
plt.show();

print('probabilidad: ',model.predict([img_test[0:1], img_test[2:3]]));

