from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import imageio as iio
from icecream import ic

labels = pd.read_excel(
        "data.nosync/NB2_classification.xlsx",
        dtype = {"file_name": str, "shoulder_score": np.int32}
        )

images_folder = "data.nosync/images/"


images = []
for name in labels.file_name:
    img = iio.imread(images_folder+name)
    images.append(img)

X_train, X_test, y_train, y_test = train_test_split(images, labels.shoulder_score, test_size=0.15)
X_train = np.array(X_train)

ic(X_train.shape)

save_folder = "data.nosync/train/"
ctr = 0
for img, label in zip(X_train, y_train):
    filename = save_folder + str(label) + "/" + str(ctr) + ".jpg"
    iio.imwrite(filename, img)
    ctr = ctr + 1

save_folder = "data.nosync/test/"
ctr = 0
for img, label in zip(X_test, y_test):
    filename = save_folder + str(label) + "/" + str(ctr) + ".jpg"
    iio.imwrite(filename, img)
    ctr = ctr + 1
