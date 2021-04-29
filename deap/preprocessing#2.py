import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

with open("E:\\datasets\\DEAP\\generated\\xs.pkl", "rb") as file:
    xs = pickle.load(file)

with open("E:\\datasets\\DEAP\\generated\\sub_ys.pkl", "rb") as file:
    sub_ys = pickle.load(file)

del xs[27]
del sub_ys[27]

x = np.concatenate(xs, axis=0)
y = np.concatenate(sub_ys, axis=0)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=23)

kernel_size_1 = 60
kernel_size_2 = 10
drop_rate = 0.5
learning_rate = 1e-3


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv1D(8, kernel_size=kernel_size_1, input_shape=(x_train.shape[1],
                                                                            x_train.shape[2]),
                                 data_format="channels_first", activation="relu"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.AvgPool1D(pool_size=2))

model.add(tf.keras.layers.Conv1D(16, kernel_size=kernel_size_2,data_format="channels_first",
                                 activation="relu"))

model.add(tf.keras.layers.SpatialDropout1D(rate=drop_rate))

model.add(tf.keras.layers.AvgPool1D(pool_size=2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dropout(rate=drop_rate))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dropout(rate=drop_rate))
model.add(tf.keras.layers.Dense(2, activation="linear"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.mean_absolute_error, metrics=["mae", "mse"])

model.summary()


model.fit(x_train, y_train, batch_size=5, epochs=50, validation_split=0.2)



model.evaluate(x_test, y_test)