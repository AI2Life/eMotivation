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


raw_x_train, x_test_raw, raw_y_train, y_test_raw = train_test_split(x, labels, random_state=23,
                                                         stratify=labels)

y_train = tf.keras.utils.to_categorical(raw_y_train)
y_test = tf.keras.utils.to_categorical(y_test_raw)




import os
import pickle
TRAIN_PATH = "D:\\datasets\\deapdataset\\generated\\train"
TEST_PATH = "D:\\datasets\\deapdataset\\generated\\test"
counter = 0

for x, y in zip(raw_x_train, y_train):
    counter += 1
    with open(os.path.join(TRAIN_PATH, str(counter) + ".pkl"), "wb") as file:
        pickle.dump((x,y), file)

counter = 0
for x, y in zip(x_test_raw, y_test):
    counter += 1
    with open(os.path.join(TEST_PATH, str(counter) + ".pkl"), "wb") as file:
        pickle.dump((x,y), file)
