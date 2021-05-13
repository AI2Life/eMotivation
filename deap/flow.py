import tensorflow as tf
import os
import pickle
import numpy as np

TRAIN_PATH = "D:\datasets\deapdataset\generated\\train"
TEST_PATH = "D:\datasets\deapdataset\generated\\test"


def gen_train():
    for p in os.scandir(TRAIN_PATH):
        with open(p.path, "rb") as file:
            data = pickle.load(file)
        yield data[0], data[1]

def gen_test():
    for p in os.scandir(TEST_PATH):
        with open(p.path, "rb") as file:
            data = pickle.load(file)
        yield data[0], data[1]


train = tf.data.Dataset.from_generator(gen_train,
                                       output_types=(np.float64, np.float64),
                                       output_shapes=((32, 30720),(3,)))

test = tf.data.Dataset.from_generator(gen_test,
                                       output_types=(np.float64, np.float64),
                                       output_shapes=((32, 30720),(3,)))
#
# x = train.as_numpy_iterator()
# for a in x:
#     print(a[0].shape, a[1].shape)

#%%
kernel_size_1 = 512*5
kernel_size_2 = 60
drop_rate = 0.4
learning_rate = 1e-4


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv1D(16, kernel_size=kernel_size_1, input_shape=(32, 30720),
                                 data_format="channels_first", activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.AvgPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(16, kernel_size=kernel_size_2,data_format="channels_first",
                                 activation="relu"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.SpatialDropout1D(rate=drop_rate))

model.add(tf.keras.layers.AvgPool1D(pool_size=2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(32, activation="relu"))

model.add(tf.keras.layers.Dropout(rate=drop_rate))

model.add(tf.keras.layers.Dense(3, activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])

model.summary()


model.fit(x=train.batch(10), epochs=100)