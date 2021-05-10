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
del xs
del sub_ys

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(y)
centroid = kmeans.cluster_centers_
labels = kmeans.labels_


raw_x_train, x_test, raw_y_train, y_test = train_test_split(x, labels, random_state=23,
                                                         stratify=labels)

import time
start = time.time()
from imblearn.over_sampling import SMOTE
sampling_rate = {0:1000, 1:1000, 2:1000}
sm = SMOTE(random_state=42, sampling_strategy=sampling_rate)
sm_x_train, y_train = sm.fit_resample(raw_x_train.reshape(raw_x_train.shape[0],
                                      raw_x_train.shape[1]*raw_x_train.shape[2]), raw_y_train)

x_train = sm_x_train.reshape(sm_x_train.shape[0], int(sm_x_train.shape[1]/32), 32)

print("smote took: ", time.time() - start)


#%%
kernel_size_1 = 30
kernel_size_2 = 6
drop_rate = 0.4
learning_rate = 1e-4


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv1D(16, kernel_size=kernel_size_1, input_shape=(x_train.shape[1],
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
model.add(tf.keras.layers.Dense(3, activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])

model.summary()


model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))



# model.evaluate(x_test, y_test)