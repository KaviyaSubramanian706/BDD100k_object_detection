from keras_cv import bounding_box
from keras_cv import visualization
import json
import tensorflow as tf
from preprocessing import Preprocess
from yolov8_detecter import YOLOV8

MODEL_PATH = "model_val.h5"

def visualize_dataset(num_images, inputs, value_range, rows, cols, bounding_box_format, class_mapping):
    """Visualizes the ground truth dataset

    Args:
        num_images (int): number of images to be viewed
        inputs (tensorflow.python.data.ops.prefetch_op._PrefetchDataset): tf data to be viewed
        value_range (tuple[int,int]): value range of input pixels
        rows (int): rows of images to be displayed
        cols (int): _description_
        bounding_box_format (str): format of bounding box "xyxy" or "xywh"
        class_mapping (dict): dictionary of class labels to integers
    """
    for i in range(num_images):
        images,y_true = next(iter(inputs.take(i+1)))
        
        visualization.plot_bounding_box_gallery(
            images,
            value_range=value_range,
            rows=rows,
            cols=cols,
            y_true=y_true,
            scale=5,
            font_scale=0.7,
            show=True,
            bounding_box_format=bounding_box_format,
            class_mapping=class_mapping,
        )


def visualize_detections(num_images, model, dataset, bounding_box_format, class_mapping):
    """Visualize detection for trained model

    Args:
        num_images (int): number of images to be viewed
        model (YOLOV8): model object to be predicted
        dataset (tensorflow.python.data.ops.prefetch_op._PrefetchDataset): tf data to be viewed
        bounding_box_format (str): format of bounding box "xyxy" or "xywh"
        class_mapping (int): dictionary of class labels to integers
    """
    for i in range(num_images):
        images, y_true = next(iter(dataset.take(i+1)))
        y_pred = model.predict(images)
        y_pred = bounding_box.to_ragged(y_pred)
        visualization.plot_bounding_box_gallery(
            images,
            value_range=(0, 255),
            bounding_box_format=bounding_box_format,
            y_true=y_true,
            y_pred=y_pred,
            scale=4,
            rows=2,
            cols=2,
            show=True,
            font_scale=0.7,
            class_mapping=class_mapping,
        )

if __name__ == "__main__":

    # Get user input for options
    user_input = input("Enter your option:\n 1. Training data\n 2. Prediction on validation data ")
    num_images = input("Enter no of images to be viewed ")

    # Load the json file
    config = json.load(open("config.json"))

    # Create class label encoding with integers
    class_mapping = dict(zip(range(len(config["classes"])), config["classes"]))
    class_dict = dict(zip(config["classes"], range(len(config["classes"]))))
 
    # Based on user option display ground truth or predictions
    if user_input=="1":
        # Preprocess the data
        train_preprocess = Preprocess(config["train"]["data_path"], config["train"]["annot_path"],\
                        config["batch_size"], class_dict)

        train_data = train_preprocess.get_tf_data()
        augmenter = train_preprocess.augmenter()
        train_ds = train_data.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(train_preprocess.dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        visualize_dataset(int(num_images), train_ds, value_range=(0,255), rows=2,\
                           cols=2, bounding_box_format="xyxy", class_mapping=class_mapping)
        # Ground truths are in blue colour 
        # Predictions are in yellow colour

    elif user_input=="2":
        # Preprocess the data
        val_preprocess = Preprocess(config["val"]["data_path"],config["val"]["annot_path"],\
                         config["batch_size"], class_dict)
        val_data = val_preprocess.get_tf_data()
        resizing = val_preprocess.resizing()
        val_ds = val_data.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(val_preprocess.dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        # Load model weights
        yolo = YOLOV8(class_mapping)
        yolo.model.load_weights(MODEL_PATH)

        visualize_detections(int(num_images), yolo.model, val_ds, "xyxy", class_mapping)
    else:
        print("Enter a valid input")