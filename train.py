# -*- coding: utf-8 -*-
"""yolov8

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VH9DCRZfTV2bhx6KbQY5Ftir6_wySGGF
"""



import json
import tensorflow as tf
import keras_cv
from tqdm.keras import TqdmCallback
from yolov8_detecter import YOLOV8
from preprocessing import Preprocess
from callbacks import EvaluateCOCOMetricsCallback


# Open and load the config.json file get inputs
config = json.load(open("config.json"))

# Map the class names to integers
class_mapping = dict(zip(range(len(config["classes"])), config["classes"]))
class_dict = dict(zip(config["classes"], range(len(config["classes"]))))


# Preprocess the train and validation data
train_preprocess = Preprocess(config["train"]["data_path"], config["train"]["annot_path"],\
                        config["batch_size"], class_dict)
val_preprocess = Preprocess(config["val"]["data_path"], config["val"]["annot_path"],\
                         config["batch_size"], class_dict)

train_data = train_preprocess.get_tf_data()
val_data = val_preprocess.get_tf_data()

augmenter = train_preprocess.augmenter()
train_ds = train_data.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(train_preprocess.dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

resizing = val_preprocess.resizing()
val_ds = val_data.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(val_preprocess.dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Create YOLOV8 object to train
yolo = YOLOV8(class_mapping)
yolo.model.summary()

# Create optimizer object to compile
optimizer = tf.keras.optimizers.Adam(
    learning_rate=config["learning_rate"]
)

# Compile the model
yolo.model.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs_yolov8large")

# Fit the model with training and val data
# Best model is saved in EvaluateCOCOMetricsCallback

history = yolo.model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["epochs"],
    callbacks=[
        EvaluateCOCOMetricsCallback(val_ds, config["model_name"]),
        tensorboard_callback,
        TqdmCallback(verbose=2)
    ]
)
