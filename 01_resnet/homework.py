# resnet tensorflow
# import tensorflow as tf

# input_shape = (2, 3, 4)
# x1 = tf.random.normal(input_shape)
# x2 = tf.random.normal(input_shape)

# print("x1: ", x1)
# print("x2: ", x2)
# y = tf.keras.layers.Add()([x1, x2])
# print("y: ", y)
# print(y.shape)
# print(type(y))

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import * 

class SimpleResNet(tf.keras.Model):
    def __init__(self):

        super(SimpleResNet, self).__init__()

        self.relu = Activation('relu')

        self.conv0 = Sequential([
            Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization(),
            Activation('relu')
        ])

        self.block11 = Sequential([
            Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization()
        ])

        self.block12 = Sequential([
            Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization()
        ])

        self.conv2 = Conv2D(filters=32, kernel_size=3, padding='same', strides=2, use_bias=False)

        self.block21 = Sequential([
            Conv2D(filters=32, kernel_size=3, padding='same', strides=2, use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization()
        ])

        self.block22 = Sequential([
            Conv2D(filters=32, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization()
        ])

        self.conv3 = Conv2D(filters=64, kernel_size=3, padding='same', strides=2, use_bias=False)

        self.block31 = Sequential([
            Conv2D(filters=64, kernel_size=3, padding='same', strides=2, use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters=64, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization()
        ])

        self.block32 = Sequential([
            Conv2D(filters=64, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(filters=64, kernel_size=3, padding='same', strides=1, use_bias=False),
            BatchNormalization()
        ])

        self.avg_pool = AveragePooling2D(pool_size=(8, 8))

        self.flatten = Flatten()

        # self.batchnorm = BatchNormalization(trainable=True)
        # self.relu = Activation('relu')
        self.fc = Dense(10)
        self.softmax = Activation('softmax')



model = SimpleResNet().model()

model.load_weights('/home/pej/Study/cifar10_model.data-00000-of-00001')

print("load success!")



        res11 = out0
        out11 = self.block11(out0)
        out11 += res11
        # out11 = Add()([out11, res11])
        out11 = self.relu(out11)

        res12 = out11
        out12 = self.block12(out11)
        out12 += res12
        out12 = self.relu(out12)

        res21 = self.conv2(out12)

        out21 = self.block21(out12)
        out21 += res21
        out21 = self.relu(out21)

        res22 = out21
        out22 = self.block22(out21)
        out22 += res22
        out22 = self.relu(out22)

        res31 = self.conv3(out22)
        out31 = self.block31(out22)
        out31 += res31
        out31 = self.relu(out31)


        res31 = out31
        out32 = self.block32(out31)
        out32 += res31
        out32 = self.relu(out32)

        out4 = self.avg_pool(out32)
        out4 = self.flatten(out4)
        out4 = self.fc(out4)
        outputs = self.softmax(out4)

        return outputs

    def model(self):
        inputs = Input(shape=(32, 32, 3))
        outputs = self.call(inputs)

        return Model(inputs=inputs, outputs=outputs)



model = SimpleResNet()
model.model().summary()


batch_size = 32
epochs = 3

loss_fn = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=0.001)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

def preprocess(x, y):
    image = tf.reshape(x, [32, 32, 3])
    image = tf.cast(image, tf.float32) / 255.0 #이미지를 float32 데이터 타입으로 바꾸고 /255.0
    image = (image - 0.5) / 0.5

    label = tf.one_hot(y, depth=10) # one hot를 10개 depth
    label = tf.squeeze(label) # [1,10] -> [10,] 해야 밑에 loss에서 계산이 됨?

    return image, label

train_loader = train_dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).repeat(3).batch(32)
# 총 6만개 전체 셔플, epoch 마다 바꿔주겠다 (defalut = False), 6만개*3=18만개 가지고 돌림

valid_loader = valid_dataset.map(preprocess).repeat(3).batch(32)
# shuffle 안함

for epoch in range(1, epochs + 1): 
    for img, label in train_loader:
        model_params = model.trainable_variables # train해야되는 param만 뽑음

        with tf.GradientTape() as tape: #pytorch는 Tape 저절로 생김
            out = model(img)
            loss = loss_fn(out, label)

        grads = tape.gradient(loss, model_params) # loss.backward
        optimizer.apply_gradients(zip(grads, model_params)) #optimizer.step()

    print(f"[{epoch}/{epochs}] finished")
    print('==================')

model.save_weights('cifar10_model', save_format='tf')

print('model saved')
