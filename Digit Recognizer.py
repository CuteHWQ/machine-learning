import numpy as np
import pandas as pd
import random
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import itertools
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 全局变量
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUBMISSION_PATH = "F:\机器学习\DigitRecongnizer\sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

TARGET = "label"
SUBMIT_TARGET = "Label"
TEST_SIZE = 0.2
LABEL_NUM = 10

IMAGE_SIZE_X = 28
IMAGE_SIZE_Y =28
SCALE_SIZE = 255

# 负载数据
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


# 分割数据(输入数据和目标数据)
y = train[TARGET]
X = train.drop([TARGET],axis=1)
X_test = test

# 缩放与变换
# 归一化数据，让CNN更快
X = X/SCALE_SIZE
X_test = X_test/SCALE_SIZE

# Reshape 图片为 3D array (height = 28px, width = 28px , canal = 1)
X = X.values.reshape(-1,IMAGE_SIZE_X,IMAGE_SIZE_Y,1)
X_test = X_test.values.reshape(-1,IMAGE_SIZE_X,IMAGE_SIZE_Y,1)

# 把label转换为one hot vectors
y = to_categorical(y, num_classes=LABEL_NUM)

# 拆分数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=TEST_SIZE,random_state=SEED )

# 模型变量
BATCH_SIZE = 64
EPOCH = 5
LEARNING_RATE = 0.001

KERNEL_INITIALIIZERS = tf.keras.initializers.GlorotNormal(seed=SEED)
PADDING = "Same"
MID_ACTIVATION = "relu"
LAST_ACTIVATION = "softmax"

COMPILE_LOSS = "categorical_crossentropy"
COMPILE_METRICS = ['accuracy']
RHO = 0.9
EPSILON = 1e-08
DROPOUT = 0.25
VERBOSE = 1

# 定义模型
def define_model():
    model = Sequential()

    model.add(Input(shape=(IMAGE_SIZE_X, IMAGE_SIZE_Y, 1)))
    model.add(Conv2D(32, kernel_size=(5, 5), kernel_initializer=KERNEL_INITIALIIZERS,
                     padding=PADDING, activation=MID_ACTIVATION))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT, seed=SEED))

    model.add(Conv2D(64, kernel_size=(5, 5), kernel_initializer=KERNEL_INITIALIIZERS,
                     padding=PADDING, activation=MID_ACTIVATION))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(DROPOUT, seed=SEED))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=KERNEL_INITIALIIZERS, activation=MID_ACTIVATION))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
    model.add(Dense(LABEL_NUM, kernel_initializer=KERNEL_INITIALIIZERS, activation=LAST_ACTIVATION))

    decay = 5 * LEARNING_RATE / EPOCH
    optimizer = RMSprop(learning_rate=LEARNING_RATE, rho=RHO, epsilon=EPSILON, decay=decay)

    model.compile(optimizer=optimizer, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)
    return model


model = define_model()
model.summary()

# 通过数据增强来防止过度拟合
train_datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False
                            )
train_generator = train_datagen.flow(X_train, y_train,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)

val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(X_val, y_val,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

reduceLROnPlateau = ReduceLROnPlateau(monitor='val_accuracy',
                                patience=3,
                                verbose=VERBOSE,
                                factor=0.5,
                                min_lr=0.00001)

# 训练模型
history = model.fit(train_generator,
                    epochs= EPOCH,
                    validation_data=val_generator,
                    verbose=VERBOSE,
                    callbacks=[reduceLROnPlateau]
                   )

# 用模型预测测试数据目标
results = model.predict(X_test)

# 把one-hot vector转换为数字
results = np.argmax(results,axis = 1)
results[:5]

# 保存最终的结果
sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[SUBMIT_TARGET] = results
sub.to_csv(SUBMISSION_PATH,index=False)
print(sub.head())

