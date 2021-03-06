import tensorflow as tf
from tensorflow.keras.layers import *

class simpleModel(tf.keras.Model):
    def __init__(self):
        super(simpleModel, self).__init__()

        self.conv = Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False)


        self.batchnorm = BatchNormalization(trainable=True)
        self.relu = Activation('relu')
        self.avg_pool = AveragePooling2D(pool_size=(8, 8))
        self.flatten = Flatten()
        self.fc = Dense(10) 
        self.softmax = Activation('softmax')

    def call(self, x):
        out0 = self.conv(x)
        out1 = self.batchnorm(out0)
        out2 = self.relu(out1)
        out3 = self.avg_pool(out2)
        out4 = self.flatten(out3)
        out5 = self.fc(out4)
        outputs = self.softmax(out5)

        return outputs

model = simpleModel()


batch_size = 32
epochs = 3

loss_fn = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=0.001)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# 전처리 
def preprocess(x, y):
    image = tf.reshape(x, [32, 32, 3])
    image = tf.cast(image, tf.float32) / 255.0 # 0 ~ 1
    image = (image-0.5) / 0.5 # -1 ~ 1

    label = tf.one_hot(y, detph=10) # -> [1, 10]
    label = tf.squeeze(label) # [1, 10] -> [10,]

    return image, label

# cifar10 이미지 전체 전부 shuffle, epoch마다 바꿔 주겠다 
train_loader = train_dataset.map(preprocess).shuffle(60000, reshuffle_each_iterations=True).repeat(3).batch(32) # 60000 * 3 

# valid_loader는 shuffle 안 한다 
valid_loader = train_dataset.map(preprocess).repeat(3).batch(32)


for epoch in range(1, epochs + 1):
    for img, label in train_loader:

        # train 해야 하는 parameter만 뽑아 놓고 
        model_params = model.trainable_variables
        
        # 파이토치는 tape가 자동으로 생성되지만 텐서플로우는 해 줘야 함 
        with tf.GradientTape() as tape:
            out = model(img)
            loss = loss_fn(out, label)

        grads = tape.gradient(loss, model_params)
        optimizer.apply_gradients(zip(grads, model_params))

    print(f"[{epoch}/{epochs}] finished")
    print('==========')

model.save_weights('cifar10_model', save_format='tf')

print('model saved')